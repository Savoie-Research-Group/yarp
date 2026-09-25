import subprocess
import psutil
from pathlib import Path
import re

class BaseJobManager:
    def submit(self, script_path: str, task_config=None) -> str: raise NotImplementedError
    def is_running(self, job_id: str) -> bool: raise NotImplementedError

class LocalJobManager(BaseJobManager):
    def submit(self, script_path: str, task_config=None) -> str:
        # Runs the bash script locally and detaches it
        process = subprocess.Popen(["bash", str(script_path)])
        return str(process.pid)

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        return psutil.pid_exists(int(job_id))

class SlurmJobManager(BaseJobManager):
    def submit(self, script_path: str, task_config=None) -> str:
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[-1]

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        try:
            result = subprocess.run(["squeue", "-j", str(job_id), "-h"], capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

class SGEJobManager(BaseJobManager):
    def submit(self, script_path: str, task_config=None) -> str:
        # Assuming qsub returns the job ID directly
        result = subprocess.run(["qsub", str(script_path)], capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[2]

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        try:
            # qstat returns exit code 0 if job is in queue/running, non-zero if finished/not found
            result = subprocess.run(["qstat", "-j", str(job_id)], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
        
class CondorJobManager(BaseJobManager):
    """
    HTCondor backend for shared-filesystem installations.

    The generated submit description references the executable and log paths
    using absolute paths. Therefore, these paths must be visible from both the
    submit node and the execute node.

    This backend does not implement HTCondor file transfer, OSDF/Pelican data
    delivery, or OSG worker-scratch orchestration.
    """
    def __init__(self, job_config=None):
        self.job_config = job_config

    def submit(self, script_path: str, task_config=None) -> str:
        submit_path = self.write_submit_file(script_path, task_config)
        result = subprocess.run(["condor_submit", str(submit_path)], capture_output=True, text=True, check=True)

        match = re.search(r"submitted to cluster\s+(\d+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse condor_submit output: {result.stdout.strip()}")

        return match.group(1)

    def is_running(self, job_id: str) -> bool:
        if not job_id: return False
        try:
            result = subprocess.run(["condor_q", str(job_id), "-autoformat", "ClusterId"], capture_output=True, text=True)
            return bool(result.stdout.strip())
        except FileNotFoundError:
            return False

    def write_submit_file(self, script_path: str, task_config=None) -> Path:
        """Write a Condor submit file for the given script."""
        script_path = Path(script_path).resolve()
        submit_path = script_path.with_suffix(".submit")
        script_path.chmod(script_path.stat().st_mode | 0o111)

        cpus = getattr(task_config, "n_cpus", 1)
        mem = getattr(task_config, "mem_per_cpu", None)
        disk = getattr(self.job_config, "request_disk", None)
        log_dir = Path(getattr(self.job_config, "log_dir", script_path.parent))
        if not log_dir.is_absolute():
            log_dir = script_path.parent / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        get_env = getattr(self.job_config, "getenv", True)
        notification = getattr(self.job_config, "notification", None) or "NEVER"
        condor_universe = getattr(self.job_config, "condor_universe", None) or "vanilla"

        with open(submit_path, "w") as f:
            f.write(f"universe = {condor_universe}\n")
            f.write(f"executable = {script_path}\n")
            if get_env:
                f.write("getenv = True\n")
            if notification:
                f.write(f"notification = {notification}\n")
            f.write(f"request_cpus = {cpus}\n")
            if mem:
                f.write(f"request_memory = {int(mem) * int(cpus)}MB\n")
            if disk:
                f.write(f"request_disk = {disk}\n")
            f.write(f"output = {log_dir / (script_path.stem + '.$(Cluster).$(Process).out')}\n")
            f.write(f"error = {log_dir / (script_path.stem + '.$(Cluster).$(Process).err')}\n")
            f.write(f"log = {log_dir / (script_path.stem + '.$(Cluster).log')}\n")
            f.write("queue 1\n")
        return submit_path

def get_job_manager(scheduler_type: str, job_config=None) -> BaseJobManager:
    """Return a job manager for the specified scheduler type."""
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "slurm":
        return SlurmJobManager()
    elif scheduler_type == "sge":
        return SGEJobManager()
    elif scheduler_type == "local":
        return LocalJobManager()
    elif scheduler_type == "condor":
        return CondorJobManager(job_config)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
