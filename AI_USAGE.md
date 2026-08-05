# AI usage in this repository

This repository is a project input for my, Nora Færevaag Solberg's, Imperial College London
Individual Research Project (IRP) - the synthetic DHI (Direct Hydrocarbon Indicator)
injection pipeline used to generate labelled training/test data for a DHI detection
model. F3 does contain real, expert-picked DHI examples (see `real_examples.py` /
`real_dhi_extraction.ipynb`), but far too few, and with no severity range, to train or
validate a detector on their own - F3 has no confirmed gas discovery, so the real
"positives" are interpreted shallow-gas anomalies, not drilling-confirmed reservoirs.
Synthetic injection provides the volume and the severity continuum those real examples
can't. This repository is not itself the graded IRP submission (`irp-nfs25`); its
outputs (the synthetic dataset) feed into that project.

## Tool used

**Claude Code** (Claude Sonnet 5), by Anthropic (https://claude.com/claude-code) - used
as the primary implementation tool throughout this repository's development: writing
and modifying the injection pipeline, dataset generation, attribute computation, the
test suite, debugging, and the fixes/write-up documents in this repo.

**Scope note:** "attribute computation" here means only `attributes.py`'s 8-channel
`compute_attribute_stack` (envelope, instantaneous phase/frequency, RMS amplitude,
sweetness, band ratio, local variance), which feeds the exported synthetic dataset. This
is separate from and much smaller than the ~20-attribute library used for detection
(coherence, spectral bandwidth, GLCM texture, dip magnitude, apparent polarity, etc.),
which lives in `irp-nfs25` and was implemented by me.

## What was AI-implemented vs. mine

Consistent with guidance confirmed by the IRP course leader Marijan Beg (AI-generated synthetic
training data is a permitted project input provided the design and validation are the
student's own intellectual contribution, and the AI is used as an implementation tool
rather than a source of decisions):

**AI-implemented (Claude Code):** the Python implementation in `src/dhi_pipeline/` and
`tests/` - wedge-model construction, the flat-spot/polarity-reversal/sag injection
mechanics, dataset generation and caching, attribute computation, SEG-Y export, and the
pytest regression suite. Also used for debugging (e.g. diagnosing OneDrive sync/
dehydration issues, a per-scenario crash exposed by the F4 footprint change) and for
drafting the fixes/write-up documents (`fixes-*.md`) from my findings and decisions.

**My own intellectual contribution:** the choice to model DHIs via a wedge-tuning
approach at all; which physical effects to include (flat spot, polarity reversal, sag/
pull-down) and why, grounded in the DHI literature (e.g. Nanda 2021); calibrating
reflection coefficients from real F3 well logs rather than accepting illustrative
defaults (including catching and fixing a coal-contamination bug and a units mismatch in
that calibration myself); the severity-tier design and thresholds; deciding what counts
as a legitimate hard negative and why (`no_conformance`, `syncline`, `single_reflector`,
`tuning`) rather than accepting a generic negative-sampling scheme; reviewing, running,
and interpreting every fix against the colleague pipeline review, including
independently reproducing the review's own numerical checks rather than taking them on
trust; the F4 footprint-scale decision and its trade-offs; the decision (and the
reasoning behind it, verified via grouped cross-validation) to revert the detector
feature set from 24 back to 12 features; and all correspondence, decisions, and
judgement calls made with my collaborator (Aziz) and supervisor throughout.

I reviewed and verified all AI-implemented code before accepting it, and can
explain the functionality of every component. The git commit history of this repository
records the iterative process in detail (commit messages describe what changed and why
for each step).
