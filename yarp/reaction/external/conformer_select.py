"""
Conformer pair selection for GSM input generation.

This class owns the pre-GSM conformer selection workflow. Joint optimization is
one strategy it calls while constructing candidate reactant/product pairs.
"""

import copy
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation

from yarp.reaction.conf_sampling.indicator import return_indicator
from yarp.reaction.external.joint_opt import make_joint_optimization_engine


class ConformerPairSelector:
    """Select reactant/product conformer pairs for GSM."""

    def __init__(self, rxn, config, scratch_dir=None, log=None):
        self.rxn = rxn
        self.config = config
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else Path.cwd() / "joint_opt"
        self.mode = config.joint_opt.lower()
        self.joint_optimizer = make_joint_optimization_engine(config, self.scratch_dir)
        self.verbose = (
            getattr(config, "verbose", False)
            or os.environ.get("YARP_DEBUG_PREGSM") == "1"
        )
        self.log = log
        self.dropped_pairs = 0
        self.model_path = None
        self.model_sha256 = None

    def select(self):
        if os.environ.get("YARP_PREGSM_CONTAINER") != "1":
            raise RuntimeError(
                "Conformer pair selection must run inside the jo_opt pre-GSM container. "
                "Use the Pysisyphus TS guess workflow instead of calling ConformerPairSelector.select() "
                "from the host YARP environment."
            )

        self._log(
            "starting selection "
            f"mode={self.mode} engine={getattr(self.config, 'joint_opt_engine', '')} "
            f"n_conf={self.config.n_conf} scratch={self.scratch_dir}"
        )

        r_confs = self._get_conformers(self.rxn.reactant)
        p_confs = self._get_conformers(self.rxn.product)
        self._log(
            f"input conformers reactant={len(r_confs)} product={len(p_confs)} "
            f"reactant_types={self._conf_types(r_confs)} product_types={self._conf_types(p_confs)}"
        )
        biased_r, biased_p = self._generate_biased_endpoints(r_confs, p_confs)
        model = self._load_model(r_confs, p_confs)

        r_confs, p_confs, biased_r, biased_p = self._truncate_conformer_pool(
            r_confs,
            p_confs,
            biased_r,
            biased_p,
        )
        self._log(
            f"post-truncation conformers reactant={len(r_confs)} product={len(p_confs)} "
            f"biased_reactant={len(biased_r)} biased_product={len(biased_p)}"
        )

        approved_pairs = []
        approved_indicators = []

        self._score_reactant_seeded_pairs(r_confs, biased_p, model, approved_pairs, approved_indicators)
        self._score_product_seeded_pairs(p_confs, biased_r, model, approved_pairs, approved_indicators)

        approved_pairs.sort(key=lambda x: x["score"], reverse=True)

        self._log(f"approved_pairs={len(approved_pairs)} dropped_pairs={self.dropped_pairs}")
        for ind, pair in enumerate(approved_pairs[:self.config.n_conf], start=1):
            self._log(
                f"selected_pair={ind} score={float(pair.get('score', 0.0)):.6f} "
                f"r_conf={getattr(pair.get('r_conf'), 'type', '')} "
                f"p_conf={getattr(pair.get('p_conf'), 'type', '')}"
            )

        return approved_pairs[:self.config.n_conf]

    def _log(self, message):
        if self.verbose:
            print(f"[ConformerPairSelector] {message}", file=self.log, flush=True)

    @staticmethod
    def _get_conformers(state):
        return [conf for key, conf in state.conformers.items() if key != "initial_geom"]

    @staticmethod
    def _conf_types(confs):
        return [getattr(conf, "type", "") for conf in confs]

    def _generate_biased_endpoints(self, r_confs, p_confs):
        biased_r = r_confs
        biased_p = p_confs

        # biased_p: product geometries guided by reactant geometries
        # biased_r: reactant geometries guided by product geometries
        if self.mode in ["dual", "r_only"]:
            biased_p = [
                self.joint_optimizer.optimize(c, self.rxn.reactant.paired_bem, f"r_to_p_{ind:04d}")
                for ind, c in enumerate(r_confs)
            ]
        if self.mode in ["dual", "p_only"]:
            biased_r = [
                self.joint_optimizer.optimize(c, self.rxn.product.paired_bem, f"p_to_r_{ind:04d}")
                for ind, c in enumerate(p_confs)
            ]

        self._log(
            "biased endpoints "
            f"reactant_none={sum(_ is None for _ in biased_r)} "
            f"product_none={sum(_ is None for _ in biased_p)}"
        )
        return biased_r, biased_p

    def _load_model(self, r_confs, p_confs):
        n_conf = self.config.n_conf
        total_conf = len(r_confs) + len(p_confs)

        self._log(f"total reactant+product conformers={total_conf}")

        model_dir = Path(__file__).resolve().parents[1] / "conf_sampling"
        if total_conf / n_conf > 3.0:
            model_path = model_dir / "rich_model.sav"
        else:
            model_path = model_dir / "poor_model.sav"

        if not model_path.exists():
            raise FileNotFoundError(
                "Conformer-pair selection model is missing. "
                f"Expected {model_path}. Rebuild the jo_opt container with the required model artifacts."
            )

        self.model_path = str(model_path)
        self.model_sha256 = self._sha256(model_path)
        self._log(
            f"model={model_path.stem} path={model_path} sha256={self.model_sha256}"
        )
        with open(model_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _truncate_conformer_pool(self, r_confs, p_confs, biased_r, biased_p):
        limit = self.config.n_conf * 5
        if len(r_confs) > limit:
            r_confs = r_confs[:limit]
            biased_p = biased_p[:limit]
        if len(p_confs) > limit:
            p_confs = p_confs[:limit]
            biased_r = biased_r[:limit]
        return r_confs, p_confs, biased_r, biased_p

    def _score_reactant_seeded_pairs(self, r_confs, biased_p, model, approved_pairs, approved_indicators):
        if self.mode in ["dual", "r_only", "off"]:
            r_loop_confs = r_confs if self.mode != "off" else r_confs[:len(biased_p)]
        else:
            r_loop_confs = []

        for ind, r_c in enumerate(r_loop_confs):
            if ind >= len(biased_p) or biased_p[ind] is None:
                self.dropped_pairs += 1
                continue

            best_p, best_ind, best_prob = self._score_pair(
                fixed_conf=r_c,
                biased_conf=biased_p[ind],
                indicator_elements=r_c.elements,
                reactant_geo=r_c.geo,
                product_geo=biased_p[ind].geo,
                model=model,
            )

            if best_prob[0][1] > 0.0 and check_uniqueness(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": r_c,
                    "p_conf": best_p,
                    "score": best_prob[0][1],
                })
            else:
                self.dropped_pairs += 1

    def _score_product_seeded_pairs(self, p_confs, biased_r, model, approved_pairs, approved_indicators):
        p_loop_confs = p_confs if self.mode in ["dual", "p_only"] else []

        for ind, p_c in enumerate(p_loop_confs):
            if ind >= len(biased_r) or biased_r[ind] is None:
                self.dropped_pairs += 1
                continue

            best_r, best_ind, best_prob = self._score_pair(
                fixed_conf=p_c,
                biased_conf=biased_r[ind],
                indicator_elements=p_c.elements,
                reactant_geo=biased_r[ind].geo,
                product_geo=p_c.geo,
                model=model,
            )

            if best_prob[0][1] > 0.0 and check_uniqueness(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": best_r,
                    "p_conf": p_c,
                    "score": best_prob[0][1],
                })
            else:
                self.dropped_pairs += 1

    @staticmethod
    def _score_pair(fixed_conf, biased_conf, indicator_elements, reactant_geo, product_geo, model):
        ind_unaligned = return_indicator(E=indicator_elements, RG=reactant_geo, PG=product_geo)
        prob_unaligned = model.predict_proba(ind_unaligned)

        aligned_biased = align_conformers(fixed_conf, biased_conf)
        if reactant_geo is biased_conf.geo:
            aligned_reactant_geo = aligned_biased.geo
            aligned_product_geo = product_geo
        else:
            aligned_reactant_geo = reactant_geo
            aligned_product_geo = aligned_biased.geo

        ind_aligned = return_indicator(E=indicator_elements, RG=aligned_reactant_geo, PG=aligned_product_geo)
        prob_aligned = model.predict_proba(ind_aligned)

        if prob_aligned[0][1] > prob_unaligned[0][1]:
            return aligned_biased, ind_aligned, prob_aligned
        return biased_conf, ind_unaligned, prob_unaligned


def align_conformers(conf, biased_conf):
    """
    Uses ASE to minimize rotation and translation between conformers.
    Returns a NEW conformer with aligned coordinates.
    """
    atoms = Atoms(symbols=[el.upper() for el in conf.elements], positions=conf.geo)
    biased_atoms = Atoms(symbols=[el.upper() for el in conf.elements], positions=biased_conf.geo)

    minimize_rotation_and_translation(atoms, biased_atoms)

    aligned_biased_conf = copy.deepcopy(biased_conf)
    aligned_biased_conf.geo = biased_atoms.positions
    aligned_biased_conf.type = f"aligned_{biased_conf.type}"

    return aligned_biased_conf


def check_uniqueness(new_indicators, approved_indicators_list, threshold=0.025):
    """Checks if a pair is too geometrically similar to an already approved pair."""
    if len(approved_indicators_list) == 0:
        return True

    min_dis = min([np.linalg.norm(np.array(new_indicators) - np.array(j)) for j in approved_indicators_list])
    return min_dis > threshold
