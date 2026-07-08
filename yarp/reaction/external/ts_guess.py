import os
import shutil
import fnmatch
import pickle
import json
import subprocess
import traceback
import signal
import time
from pathlib import Path

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer
from yarp.reaction.external.conformer_select import ConformerPairSelector

class TSGuessTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        r_node = self.rxn.reactant
        p_node = self.rxn.product
        if not r_node.conformers or not p_node.conformers:
            return False

        r_keys = r_node.conformers.keys()
        r_match = False
        for rk in r_keys:
            if 'conf_gen' in rk and r_node.conformers[rk].geo is not None: 
                r_match = True
        p_keys = p_node.conformers.keys()
        p_match = False
        for pk in p_keys:
            if 'conf_gen' in pk and p_node.conformers[pk].geo is not None:
                p_match = True

        if not r_match or not p_match:
            return False

        return True

class PysisyphusTSGuessCalculator(TSGuessTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:jo_opt"
        self.n_pairs = self.config.n_conf

    def generate_input(self):
        """
        Writes the payload consumed by the container-side pre-GSM workflow.

        Conformer selection, joint optimization, ML scoring, and Pysisyphus GSM
        execution all run inside the jo_opt container so the host YARP
        environment does not need the ML dependency stack.
        """
        payload_path = self.scratch_dir / "payload.pkl"
        with open(payload_path, "wb") as f:
            pickle.dump({"rxn": self.rxn, "config": self.config}, f)

    def write_submission_script(self):
        """
        Creates a sequential runner script for inside the container, 
        and a host submission script to launch the container.
        """
        # ---------------------------------------------------------
        # 1. The INNER Script (Runs inside the Docker container)
        # ---------------------------------------------------------
        inner_script_path = self.scratch_dir / "run_pysis_inner.sh"
        
        with open(inner_script_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("set -u\n")
            f.write("trap 'ec=$?; echo \"Inner GSM wrapper exiting with code ${ec}\"' EXIT\n")
            f.write("echo 'Starting YARP serial GSM execution...'\n\n")

            f.write("export PYSISRC=/root/.pysisyphusrc\n")
            f.write("export YARP_PREGSM_CONTAINER=1\n")
            f.write("ulimit -s unlimited || true\n")
            for key, value in self.thread_env(self.config.n_cpus).items():
                f.write(f"export {key}={value}\n")
            f.write(
                "python - <<'PY'\n"
                "from pathlib import Path\n"
                "import yarp.reaction.external.ts_guess as ts_guess\n"
                "import yarp.reaction.conf_sampling.indicator as indicator\n"
                "print(f'Container ts_guess module: {Path(ts_guess.__file__).resolve()}', flush=True)\n"
                "model_dir = Path(indicator.__file__).resolve().parent\n"
                "print(f'Container conf_sampling dir: {model_dir}', flush=True)\n"
                "print(f'Container selector models: {[p.name for p in model_dir.glob(\"*.sav\")]}', flush=True)\n"
                "PY\n"
            )
            f.write(
                "python -c \""
                "from yarp.reaction.external.ts_guess import PysisyphusTSGuessCalculator; "
                "PysisyphusTSGuessCalculator.run_containerized('/work/payload.pkl')"
                "\"\n"
            )
                
        # Make executable
        inner_script_path.chmod(0o755)

        # ---------------------------------------------------------
        # 2. The OUTER Script (Runs on the Host / Job Manager)
        # ---------------------------------------------------------
        submit_script_path = self.scratch_dir / "submit_gsm.sh"
        
        # Pulls the docker prefix (e.g., 'docker run --rm -v /scratch:/work -u UID:GID yarp_pysisyphus')
        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))
        
        with open(submit_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            self.write_scheduler_headers(f)
            self.write_thread_env(f)
            # Launch the container and tell it to run the inner script
            f.write(f"cd {self.scratch_dir}\n")
            f.write("echo \"$(date) starting containerized GSM wrapper\" > container_lifecycle.log\n")
            f.write(
                "trap 'ec=$?; echo \"$(date) submit_gsm.sh exiting with code ${ec}\" "
                ">> container_lifecycle.log' EXIT\n"
            )
            f.write(f"{prefix} bash /work/run_pysis_inner.sh > container.out 2> container.err\n")
            
        submit_script_path.chmod(0o755)
        
        # Return the outer script for the Job Manager to execute
        return str(submit_script_path)

    def check_output(self) -> bool:
        """
        Returns True if AT LEAST ONE of the N pairs generated a TS guess.
        """
        num_runs = self._get_num_runs()
        if num_runs == 0:
            self._print_container_log_tail()
            return False # No runs found at all!

        one_successful = False
        
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"gsm_run{i}"
            log_file = run_dir / f"gsm_{i}.log"
            trj_file = run_dir / "final_geometries.trj"
            xyz_file = run_dir / "splined_hei.xyz"

            # 1. File existence check
            if not (log_file.exists() and trj_file.exists() and xyz_file.exists()):
                print(f"     * Run {i} failed: Missing expected output files.")
                continue

            # 2. Log file termination check
            with open(log_file, "r") as f:
                log_text = f.read()

            if "Wrote splined HEI" not in log_text or "pysisyphus run took" not in log_text:
                print(f"     ! Run {i} failed: Did not find successful termination message in log. Try increasing mem_per_cpu for tasks using 'pysisyphus'.")
                continue

            # If it passes all checks, at least one run succeeded!
            one_successful = True
            
        # We only care if at least one succeeded
        if not one_successful:
            self._print_container_log_tail()
        return one_successful

    def _print_container_log_tail(self):
        for name in ("container.err", "container.out", "pregsm.log"):
            path = self.scratch_dir / name
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            if not lines:
                continue
            print(f"     [TS Guess] Last lines of {name}:")
            for line in lines[-20:]:
                print(f"       {line}")

    def scrape_data(self) -> bool:
        """
        Validates the Pysisyphus GSM run and extracts the geometries into 
        the reaction object. 
        """
        num_runs = self._get_num_runs()
        
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"gsm_run{i}"
            log_file = run_dir / f"gsm_{i}.log"
            trj_file = run_dir / "final_geometries.trj"
            xyz_file = run_dir / "splined_hei.xyz"

            # --- SAFEGUARD: Skip failed runs during partial success ---
            if not (log_file.exists() and trj_file.exists() and xyz_file.exists()):
                continue
            with open(log_file, "r") as f:
                if "Wrote splined HEI" not in f.read():
                    continue
            # ---------------------------------------------------------

            # Parse and store the splined TS guess
            ts_elements, ts_geo = xyz_parse(xyz_file, multiple=False)
            
            ts_conf = conformer()
            ts_conf.elements = ts_elements
            ts_conf.geo = ts_geo
            ts_conf.lot = self.config.gsm_lot
            ts_conf.software = self.config.software
            ts_conf.type = f"ts_guess_{i}_{ts_conf.lot}_{ts_conf.software}"
            
            # Store in the reaction object's TS dictionary
            self.rxn.ts_geom[ts_conf.type] = ts_conf

            # Parse the final geometries trajectory for Reactant/Product pairs
            trj_elements, trj_geo = xyz_parse(trj_file, multiple=True)
            
            if len(trj_elements) >= 2:
                # Store Reactant guess (first frame)
                r_conf = conformer()
                r_conf.elements = trj_elements[0]
                r_conf.geo = trj_geo[0]
                r_conf.lot = self.config.gsm_lot
                r_conf.software = self.config.software
                r_conf.type = f"guess_conf_{i}_{r_conf.lot}_{r_conf.software}"
                
                self.rxn.reactant.conformers[r_conf.type] = r_conf
                
                # Store Product guess (last frame)
                p_conf = conformer()
                p_conf.elements = trj_elements[-1]
                p_conf.geo = trj_geo[-1]
                p_conf.lot = self.config.gsm_lot
                p_conf.software = self.config.software
                p_conf.type = f"guess_conf_{i}_{p_conf.lot}_{p_conf.software}"
                
                self.rxn.product.conformers[p_conf.type] = p_conf
        
        return True

    def cleanup(self):
        """Per run dir: keep inputs, logs, xyzs, and trajectories; delete xTB calc dirs."""
        num_runs = self._get_num_runs()
        for i in range(1, num_runs + 1):
            run_dir = self.scratch_dir / f"gsm_run{i}"
            if not run_dir.exists():
                continue

            keep_exact = {f"gsm_{i}_input.yaml", f"gsm_{i}.log", "splined_hei.xyz"}
            keep_patterns = ["*.trj", "*.xyz"]

            for item in run_dir.iterdir():
                if item.name in keep_exact:
                    continue
                if item.is_file() and any(fnmatch.fnmatch(item.name, p) for p in keep_patterns):
                    continue
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def _write_pysis_gsm_input(self, input_path, r_xyz_path, p_xyz_path):
        self.write_pysis_gsm_input(input_path, r_xyz_path, p_xyz_path, self.config)

    def _get_num_runs(self) -> int:
            """
            Dynamically counts the number of gsm_run subdirectories.
            Crucial for stateless execution where self.pairs_to_run is lost between runs.
            """
            # Finds all directories matching 'gsm_run*' in the scratch folder
            run_dirs = list(self.scratch_dir.glob("gsm_run*"))
            return len(run_dirs)

    @staticmethod
    def write_pysis_gsm_input(input_path, r_xyz_path, p_xyz_path, config):
        # Make sure lot is xTB (ERM: We'll make this more robust later! Hopefully!)
        lot = config.gsm_lot.lower()

        with open(input_path, 'w') as f:
            input_geo = [r_xyz_path, p_xyz_path]
            f.write(f'geom:\n type: cart\n fn: {input_geo}\n')

            # ERM: I left out the option for solvent,
            # because what I saw in classy YARP didn't make sense to me...
            f.write(f'calc:\n type: {lot}\n pal: {config.n_cpus}\n mem: {config.mem_per_cpu}\n charge: {config.charge}\n mult: {config.multiplicity}\n')

            f.write(f'cos:\n type: gs\n max_nodes: {config.max_gsm_nodes}\n climb: True\n climb_rms: 0.005\n climb_lanczos: False\n reparam_check: rms\n reparam_every: 1\n reparam_every_full: 1\n')

            f.write(f'opt:\n type: string\n stop_in_when_full: -1\n align: True\n scale_step: global\n')

    @staticmethod
    def _write_selected_pairs_manifest(path, pairs, selector=None):
        manifest = {
            "n_pairs": len(pairs),
            "pairs": [
                {
                    "index": ind,
                    "score": float(pair.get("score", 0.0)),
                    "r_conf_type": getattr(pair.get("r_conf"), "type", ""),
                    "p_conf_type": getattr(pair.get("p_conf"), "type", ""),
                }
                for ind, pair in enumerate(pairs, start=1)
            ],
        }
        if selector is not None:
            manifest["selector"] = {
                "joint_opt": selector.mode,
                "joint_opt_engine": getattr(selector.config, "joint_opt_engine", ""),
                "n_conf": selector.config.n_conf,
                "model_path": selector.model_path,
                "model_sha256": selector.model_sha256,
                "dropped_pairs": selector.dropped_pairs,
            }
        Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _runtime_to_seconds(runtime):
        try:
            hours, minutes, seconds = [int(_) for _ in str(runtime).split(":")]
        except Exception:
            return 3600
        return hours * 3600 + minutes * 60 + seconds

    @classmethod
    def _pair_timeout_seconds(cls, config, n_pairs):
        total = cls._runtime_to_seconds(getattr(config, "max_runtime", "01:00:00"))
        n_pairs = max(1, int(n_pairs))
        # Keep scheduler tear-down time available for logging and later pairs.
        return max(60, int(total * 0.8 / n_pairs))

    @staticmethod
    def _run_pysis(pair_dir, inp_name, out_path, err_path, timeout_s, log):
        start = time.time()
        with out_path.open("w", encoding="utf-8") as out:
            with err_path.open("w", encoding="utf-8") as err:
                proc = subprocess.Popen(
                    ["pysis", inp_name],
                    cwd=pair_dir,
                    stdout=out,
                    stderr=err,
                    text=True,
                    start_new_session=True,
                )
                print(
                    f"GSM pair in {pair_dir.name} started with pid {proc.pid}; "
                    f"timeout={timeout_s} s",
                    file=log,
                    flush=True,
                )
                deadline = start + timeout_s
                next_heartbeat = start + 60
                while True:
                    returncode = proc.poll()
                    if returncode is not None:
                        elapsed = time.time() - start
                        print(
                            f"GSM pair in {pair_dir.name} exited with code {returncode} "
                            f"after {elapsed:.1f} s",
                            file=log,
                            flush=True,
                        )
                        return returncode == 0

                    now = time.time()
                    if now >= deadline:
                        elapsed = now - start
                        print(
                            f"GSM pair in {pair_dir.name} exceeded {timeout_s} s "
                            f"after {elapsed:.1f} s; terminating and trying next pair.",
                            file=log,
                            flush=True,
                        )
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                            proc.wait(timeout=30)
                        except Exception:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
                            except Exception:
                                pass
                        return False

                    if now >= next_heartbeat:
                        elapsed = now - start
                        print(
                            f"GSM pair in {pair_dir.name} still running after {elapsed:.1f} s",
                            file=log,
                            flush=True,
                        )
                        next_heartbeat += 60

                    time.sleep(min(5, max(0.1, deadline - now)))

    @classmethod
    def run_containerized(cls, payload_path):
        """
        Container-side TS guess runner.

        This keeps pre-GSM conformer selection, joint optimization, ML pair scoring,
        and GSM execution inside the same container environment.
        """
        payload_path = Path(payload_path)
        work_dir = payload_path.parent
        pregsm_log = work_dir / "pregsm.log"

        with pregsm_log.open("w", encoding="utf-8") as log:
            def log_signal(signum, _frame):
                print(f"Received signal {signum}; aborting GSM wrapper", file=log, flush=True)
                raise RuntimeError(f"Received signal {signum}")

            handled_signals = [
                signal.SIGTERM,
                signal.SIGINT,
                signal.SIGHUP,
            ]
            for optional_signal in ("SIGUSR1", "SIGUSR2"):
                if hasattr(signal, optional_signal):
                    handled_signals.append(getattr(signal, optional_signal))

            previous_handlers = {}
            for sig in handled_signals:
                try:
                    previous_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, log_signal)
                except (OSError, RuntimeError, ValueError):
                    pass

            try:
                print("Starting containerized YARP pre-GSM selection", file=log, flush=True)

                with payload_path.open("rb") as f:
                    payload = pickle.load(f)

                rxn = payload["rxn"]
                config = payload["config"]

                os.environ["YARP_PREGSM_CONTAINER"] = "1"
                os.environ["YARP_DEBUG_PREGSM"] = "1"
                selector = ConformerPairSelector(rxn, config, scratch_dir=work_dir / "joint_opt", log=log)
                pairs = selector.select()
                cls._write_selected_pairs_manifest(work_dir / "selected_pairs.json", pairs, selector)
                print(f"Selected {len(pairs)} pair(s)", file=log, flush=True)

                if not pairs:
                    raise RuntimeError("No GSM pairs selected")

                any_success = False
                timeout_s = cls._pair_timeout_seconds(config, len(pairs))
                print(
                    f"Using per-GSM-pair timeout of {timeout_s} s",
                    file=log,
                    flush=True,
                )
                for idx, pair in enumerate(pairs, start=1):
                    pair_dir = work_dir / f"gsm_run{idx}"
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    r_xyz_path = pair_dir / f"reactant_{idx}.xyz"
                    p_xyz_path = pair_dir / f"product_{idx}.xyz"
                    inp_path = pair_dir / f"gsm_{idx}_input.yaml"

                    r_xyz_path.write_text(pair["r_conf"].to_xyz_string(), encoding="utf-8")
                    p_xyz_path.write_text(pair["p_conf"].to_xyz_string(), encoding="utf-8")
                    cls.write_pysis_gsm_input(inp_path, r_xyz_path.name, p_xyz_path.name, config)

                    print(
                        f"Running GSM pair {idx}: "
                        f"score={float(pair.get('score', 0.0)):.6f} "
                        f"r_conf={getattr(pair.get('r_conf'), 'type', '')} "
                        f"p_conf={getattr(pair.get('p_conf'), 'type', '')}",
                        file=log,
                        flush=True,
                    )
                    ok = cls._run_pysis(
                        pair_dir,
                        inp_path.name,
                        pair_dir / f"gsm_{idx}.log",
                        pair_dir / f"gsm_{idx}.err",
                        timeout_s,
                        log,
                    )
                    if ok:
                        any_success = True

                if not any_success:
                    raise RuntimeError("All GSM runs exited nonzero")
            except Exception:
                traceback.print_exc(file=log)
                log.flush()
                raise
            finally:
                for sig, previous_handler in previous_handlers.items():
                    try:
                        signal.signal(sig, previous_handler)
                    except (OSError, RuntimeError, ValueError):
                        pass
