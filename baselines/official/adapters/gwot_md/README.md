# GWOT-MD / Atanas reproduction boundary

The GWOT-MD paper is:

> Shunsuke Kamiya, Taiga Mitamura, Muneki Ikeda, and Masafumi Oizumi,
> *Unsupervised Neuronal Matching with Spontaneous Neuronal Activity*,
> ICLR 2025 Re-Align workshop.

As of the repository audit on 2026-08-30, the paper page does not link a
GWOT-MD source repository or matching caches.  The authors' lab publishes
[`oizumi-lab/GWTune`](https://github.com/oizumi-lab/GWTune), pinned here at
commit `8ad4fd949a477f7b7fb6a67205348a3a747026ec`, but that toolbox accepts one
source and one target dissimilarity matrix.  It does not implement GWOT-MD's
set of delayed distance matrices.

Consequently, results must be named carefully:

- `GWTune (official GWOT)`: the untouched authors' toolbox, applicable to
  the single-matrix `h=0` control.
- `GWOT-MD (paper-protocol reimplementation)`: the implementation in
  `baselines/gwot_md/`; this is not author-released code.
- `GWOT-MD (CV5 adaptation)`: the existing grouped-CV adapter in
  `scripts/benchmarks/run_gwot_md_atanas_cv5.py`; this uses a different cohort
  and solver search from the Appendix-F paper result.

The paper reproduction target is the Appendix-F median per-individual Top-5
accuracy of 46%, using 21 Atanas recordings, 1,000 unique nine-teacher sets,
inner leave-one-out selection of `h`, and `v=5, k=5` majority voting.  The
complete 21 x 21 common-label matrix in Figure C.1 is treated as a mandatory
denominator audit, not as a hyperparameter target.

The data preparation script beside this file builds a standalone 21-recording
NPZ input directory from the official Atanas H5 archive.  It records the source
of every label and refuses a Figure-C.1 mismatch by default.

Use `run_atanas21_paper_reproduction.sh official-h0-audit` to compare the
pinned official GWTune solver and the paper reimplementation on a real Atanas
pair at `h=0`.  The recorded audit passes to machine precision.  This validates
the single-matrix boundary and the factor-of-two GW gradient convention; it
does not turn the multi-delay implementation into author-released code.
