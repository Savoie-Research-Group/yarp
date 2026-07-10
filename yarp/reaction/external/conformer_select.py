"""
Conformer pair selection for GSM input generation.

This class owns the pre-GSM conformer selection workflow. Joint optimization is
one strategy it calls while constructing candidate reactant/product pairs.
"""

import copy
import os
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation

from yarp.reaction.conf_sampling.indicator import return_indicator
from yarp.reaction.external.joint_opt import make_joint_optimization_engine
from yarp.reaction.external.model_scorer import ContainerModelScorer
from yarp.yarpecule.graph.adjacency import table_generator


class ConformerPairSelector:
    """Select reactant/product conformer pairs for GSM."""

    def __init__(self, rxn, config, scratch_dir=None, log=None, job_manager=None):
        self.rxn = rxn
        self.config = config
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else Path.cwd() / "joint_opt"
        self.mode = config.joint_opt.lower()
        self.joint_optimizer = make_joint_optimization_engine(config, self.scratch_dir)
        self.job_manager = job_manager
        self.verbose = (
            getattr(config, "verbose", False)
            or os.environ.get("YARP_DEBUG_PREGSM") == "1"
        )
        self.log = log
        self.dropped_pairs = 0
        self.model_path = None
        self.model_sha256 = None

    def select(self):
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
        scorer = self._build_scorer(r_confs, p_confs)

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

        self._score_reactant_seeded_pairs(r_confs, biased_p, scorer, approved_pairs, approved_indicators)
        self._score_product_seeded_pairs(p_confs, biased_r, scorer, approved_pairs, approved_indicators)
        self.model_path = scorer.model_path
        self.model_sha256 = scorer.model_sha256

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

    def _build_scorer(self, r_confs, p_confs):
        n_conf = self.config.n_conf
        total_conf = len(r_confs) + len(p_confs)

        self._log(f"total reactant+product conformers={total_conf}")

        if total_conf / n_conf > 3.0:
            model_name = "rich_model"
        else:
            model_name = "poor_model"

        self._log(f"model={model_name} scorer_container={ContainerModelScorer.image_name}")
        return ContainerModelScorer(
            self.job_manager,
            self.scratch_dir / "model_scorer",
            model_name,
            log=self.log,
        )

    def _truncate_conformer_pool(self, r_confs, p_confs, biased_r, biased_p):
        limit = self.config.n_conf * 5
        if len(r_confs) > limit:
            r_confs = r_confs[:limit]
            biased_p = biased_p[:limit]
        if len(p_confs) > limit:
            p_confs = p_confs[:limit]
            biased_r = biased_r[:limit]
        return r_confs, p_confs, biased_r, biased_p

    def _score_reactant_seeded_pairs(self, r_confs, biased_p, scorer, approved_pairs, approved_indicators):
        if self.mode in ["dual", "r_only", "off"]:
            r_loop_confs = r_confs if self.mode != "off" else r_confs[:len(biased_p)]
        else:
            r_loop_confs = []

        candidates = []
        for ind, r_c in enumerate(r_loop_confs):
            if ind >= len(biased_p) or biased_p[ind] is None:
                self.dropped_pairs += 1
                continue

            candidates.append(self._build_pair_candidate(
                source_index=ind,
                fixed_conf=r_c,
                biased_conf=biased_p[ind],
                indicator_elements=r_c.elements,
                reactant_geo=r_c.geo,
                product_geo=biased_p[ind].geo,
            ))

        probabilities = self._score_candidates(candidates, scorer)
        for candidate, candidate_probs in zip(candidates, probabilities):
            best_p, best_ind, best_prob = self._select_scored_pair(candidate, candidate_probs)
            r_c = candidate["fixed_conf"]
            ind = candidate["source_index"]

            if best_prob[1] > 0.0 and check_uniqueness(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": r_c,
                    "p_conf": best_p,
                    "score": best_prob[1],
                    "source": "reactant_seeded",
                    "source_index": ind,
                    "selection_direction": "reactant_fixed_product_biased",
                    "r_graph_diffs": self._graph_diffs(r_c),
                    "p_graph_diffs": self._graph_diffs(best_p),
                })
            else:
                self.dropped_pairs += 1

    def _score_product_seeded_pairs(self, p_confs, biased_r, scorer, approved_pairs, approved_indicators):
        p_loop_confs = p_confs if self.mode in ["dual", "p_only"] else []

        candidates = []
        for ind, p_c in enumerate(p_loop_confs):
            if ind >= len(biased_r) or biased_r[ind] is None:
                self.dropped_pairs += 1
                continue

            candidates.append(self._build_pair_candidate(
                source_index=ind,
                fixed_conf=p_c,
                biased_conf=biased_r[ind],
                indicator_elements=p_c.elements,
                reactant_geo=biased_r[ind].geo,
                product_geo=p_c.geo,
            ))

        probabilities = self._score_candidates(candidates, scorer)
        for candidate, candidate_probs in zip(candidates, probabilities):
            best_r, best_ind, best_prob = self._select_scored_pair(candidate, candidate_probs)
            p_c = candidate["fixed_conf"]
            ind = candidate["source_index"]

            if best_prob[1] > 0.0 and check_uniqueness(best_ind, approved_indicators):
                approved_indicators.append(best_ind)
                approved_pairs.append({
                    "r_conf": best_r,
                    "p_conf": p_c,
                    "score": best_prob[1],
                    "source": "product_seeded",
                    "source_index": ind,
                    "selection_direction": "product_fixed_reactant_biased",
                    "r_graph_diffs": self._graph_diffs(best_r),
                    "p_graph_diffs": self._graph_diffs(p_c),
                })
            else:
                self.dropped_pairs += 1

    def _graph_diffs(self, conf):
        """Return graph differences to the intended reactant and product states."""
        try:
            adj_mat = table_generator(conf.elements, conf.geo)
            return {
                "reactant": int(np.abs(adj_mat - self.rxn.reactant.graph.adj_mat).sum()),
                "product": int(np.abs(adj_mat - self.rxn.product.graph.adj_mat).sum()),
            }
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _build_pair_candidate(source_index, fixed_conf, biased_conf, indicator_elements, reactant_geo, product_geo):
        ind_unaligned = return_indicator(E=indicator_elements, RG=reactant_geo, PG=product_geo)

        aligned_biased = align_conformers(fixed_conf, biased_conf)
        if reactant_geo is biased_conf.geo:
            aligned_reactant_geo = aligned_biased.geo
            aligned_product_geo = product_geo
        else:
            aligned_reactant_geo = reactant_geo
            aligned_product_geo = aligned_biased.geo

        ind_aligned = return_indicator(E=indicator_elements, RG=aligned_reactant_geo, PG=aligned_product_geo)
        return {
            "source_index": source_index,
            "fixed_conf": fixed_conf,
            "biased_conf": biased_conf,
            "aligned_biased": aligned_biased,
            "ind_unaligned": ind_unaligned,
            "ind_aligned": ind_aligned,
        }

    @staticmethod
    def _score_candidates(candidates, scorer):
        if not candidates:
            return []
        indicators = []
        for candidate in candidates:
            indicators.extend([candidate["ind_unaligned"], candidate["ind_aligned"]])
        scored_rows = scorer.predict_proba(indicators)
        return [scored_rows[i:i + 2] for i in range(0, len(scored_rows), 2)]

    @staticmethod
    def _select_scored_pair(candidate, candidate_probs):
        prob_unaligned, prob_aligned = candidate_probs

        if prob_aligned[1] > prob_unaligned[1]:
            return candidate["aligned_biased"], candidate["ind_aligned"], prob_aligned
        return candidate["biased_conf"], candidate["ind_unaligned"], prob_unaligned


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
