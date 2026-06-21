# YARP 3.0 oxidation-state divergence — one-page summary

*Zhao Li · 2026-06-19*

## What's going on

We rebuilt our oxidation-state (OS) extraction pipeline on the **new YARP**
refactor (classy-yarp / "YARP 3.0") and discovered it produces different
OS values than the **old YARP** version we used to publish the GoldDIGR
oxidation-state dial-plot. Several of the new YARP results were
**chemically impossible** — for example, neutral Cr-cluster complexes
reported as Cr(VI), a Pt complex assigned Pt(VII), and Cp-Ir species
shifted up by two units.

After a structured investigation, we identified three concrete bugs in
the YARP refactor, patched them locally, and verified the patched version
recovers the published behavior on hard cases while keeping every existing
YARP test passing.

## What we found — three real bugs in YARP 3.0

1. **Missing property values for 5d and 4d transition metals.** During the
   refactor, several entries in YARP's elemental property tables (valence
   electrons, polarizability, octet capacity) were left blank for atoms
   like W, Re, Os, Pt, Mo, Hf. Any reaction touching those metals crashes
   on lookup. We restored the original values from the previous YARP
   version.

2. **A redundant loop in the resonance-structure search.** The new code
   re-walks every previously found Lewis structure at each search step,
   making the algorithm ~10× slower and exploring a different distribution
   of candidate structures. Removing the redundant loop both speeds the
   pipeline up and improves agreement with the published OS numbers.

3. **A "safety-net" step that misfires on organometallic species.** The
   new YARP carries early-pass guess structures forward into the final
   metal-ligand bond classification, even when later passes correctly
   improved them. For organic molecules this is harmless and helpful.
   For transition-metal complexes (especially with Cp-type ligands) it
   lets a chemically wrong guess sneak through the metal-ligand bond
   classifier, which then drains too many electrons from the metal and
   reports a much higher oxidation state than reality. We made the
   safety-net step conditional: kept for organics, disabled when a
   transition metal is present.

We also identified one change we **shouldn't** apply (an exploratory
search-mode option that hurts both organic and TM cases) and one apparent
sign flip in the scoring function that turned out to be cosmetic — the
behavior is mathematically identical to the old version.

## What this means for GoldDIGR

- **Published OS distribution (Figure 2 of the manuscript) is still valid.**
  It was generated with the old (correctly-patched) YARP, and the
  patched-new-YARP recovers the same answers on **80.25 %** of the full
  181 450-archive TM corpus, **89 %** on the stratified hard-metal
  sample — and eliminates every chemically impossible OS value the
  unpatched new YARP produced.
- **On chemically-impossible OS counts at corpus scale, the new stack
  is about even with the old patched YARP.** It eliminates the
  catastrophic single-archive failures (Pt(VII), Cr(VI), Cp-Ir(V),
  Co(VII)) but the overall count of atoms above group-max changes by
  only +0.3 % (11 477 → 11 515). Real wins on heavy noble metals:
  Au −43 %, Ag −29 %. Other late-TM metals see small symmetric
  shifts in either direction. Dial-plot regeneration is expected to
  look similar to the published version, with slight improvements
  in the Au / Ag tails.
- **For any future OS extraction on this corpus, we'll use the patched
  YARP**, not the unmodified YARP 3.0 release.
- **The remaining ~20 % disagreement on the corpus** is not a bug
  pattern. 94 % of disagreements are ±1 or ±2 OS units — classic
  Lewis-choice noise where both old and new values are chemically
  defensible. Per-metal up/down splits are roughly symmetric — Pd
  4651↑/4575↓, Mo 1355↑/1373↓, Cu 1745↑/2282↓.

## Numbers

**Stratified sanity sample (144 reactions, 18 transition metals, weighted
toward hard rare metals where bugs were most visible):**

| | speed | match vs published OS | YARP unit tests |
|---|---:|---:|---:|
| YARP 3.0, raw | 1× | 65% | pass |
| YARP 3.0, patched (our version) | 10× | **89%** | pass |

**Full corpus run (181 450 TM reactions, completed 2026-06-20):**

| | value |
|---|---:|
| Archives processed | 181 450 |
| YARP crashes / errors | **10 (0.01 %)** |
| Full agreement with published OS values | **80.25 %** |
| Disagreements that are just ±1 or ±2 OS shifts | **94 %** of the 20 % |
| Chemically impossible OS atoms (atom-level) | OLD 11 477 → NEW 11 515 (essentially flat) |
| Reduction on heavy-noble-metal tails | Au −43 %, Ag −29 % |
| Catastrophic single-archive bugs (Pt(VII), Cr(VI), Cp-Ir(V)) | **all eliminated** |

The full-corpus number is lower than the stratified one because the
sample was deliberately biased toward hard rare metals (W, Re, Os, Pt,
Cr…) where our patches added the most lift. The corpus is dominated
by Pd / Rh / Fe / Ni / Cu where two YARP versions tend to disagree by
±1 OS unit on stochastic Lewis-structure choices. At corpus scale the
per-metal up/down split is roughly symmetric — i.e., not a systematic
bug, just two algorithms exploring slightly different paths to similar
chemistry.

## Next steps

1. **Mass-run the patched pipeline on all 181k unique TM reactions.**
   Estimated SLURM cost: ~17 hours wall on 32-way parallel; no risk to the
   archive data (read-only pipeline that writes a CSV).
2. **Send a short technical writeup to the YARP maintainers** so the three
   bugs can be fixed upstream and the wider community benefits. The
   property-table restoration and the redundant-loop removal are
   uncontroversial; the safety-net conditional needs a short design
   conversation.
3. **Optional follow-up on the residual 13%** if we decide the dial-plot
   should be regenerated at finer agreement. Two characterizable
   patterns are left (Mo-dithiolenes biased downward; first-row
   mid-OS metals biased upward); each is a ~1-day investigation.

## What we are *not* doing

- Not forking YARP. The patches live as a documented local branch and
  will be retired once upstream catches up.
- Not modifying the published archive (zip / tar.zst) files. The OS
  extraction pipeline is read-only; results go to standalone CSVs.
- Not changing the published manuscript numbers based on the patched
  YARP — those came from the old YARP and remain the reference.

---

Full technical writeup with patch details, bisection methodology, and
evidence: `YARP-3.0-OS-divergence-investigation.md` in the same directory.
