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

This repository has also grown, over the P0-P2 round of work described in
`blind_tests.md`, to contain a second DHI detector: a 14-feature ratio-based
XGBoost classifier (`ratio_features.py`, `calibration.py`, and the
`scripts/build_v2_features.py` / `train_xgb_v2.py` / `score_*.py` /
`run_blind_predictions.py` pipeline) used for cross-generator blind-exchange
testing with a collaborator. This is distinct from - and much smaller than -
the ~20-attribute detector in `irp-nfs25` described in the scope note below.

## Tool used

**Claude Code** (Claude Sonnet 5), by Anthropic
(https://claude.com/claude-code) - used as the primary implementation and debugging
tool throughout this repository's development.

**Scope note:** "attribute computation" here means only `attributes.py`'s 8-channel
`compute_attribute_stack` (envelope, instantaneous phase/frequency, RMS amplitude,
sweetness, band ratio, local variance), which feeds the exported synthetic dataset. This
is separate from and much smaller than the ~20-attribute library used for `irp-nfs25`'s
own detector (coherence, spectral bandwidth, GLCM texture, dip magnitude, apparent
polarity, etc.), which lives there and was designed, evaluated, and interpreted by the
author, with implementation assistance disclosed in that repository's `AI_USAGE.md`.
It's also separate from this repo's own 14-feature ratio-based detector (see above) -
`compute_attribute_stack` is an upstream input to that detector's features, not the
detector itself.

## Scope of AI assistance

Consistent with guidance confirmed by the IRP course leader Marijan Beg, AI-generated
synthetic training data is a permitted project input provided the design and validation
are the student's own intellectual contribution and AI is used as an implementation
tool rather than a source of decisions.

Claude Code assisted with all authored Python implementation under `scripts/`,
`src/dhi_pipeline/`, `tests/`, and `utils/`. This includes the injection pipeline,
dataset generation and caching, attribute computation, calibration, ratio-feature
extraction, model training, blind scoring, SEG-Y and supplement export, download and
verification utilities, and the regression test suite. Claude Code also assisted with
debugging, all three notebooks, and the authored Markdown documentation in this
repository. Generated datasets, model files, prediction tables, and figures are outputs
of these workflows rather than authored source material.

**OpenAI Codex** (GPT-5), by OpenAI (https://openai.com/codex/) - used to audit
this repository for missing AI acknowledgements and to help structure this updated
AI-usage disclosure. Codex was not used to implement the scientific pipeline or
detector code in this repository.

## Role of the author

Claude Code was used as an implementation and debugging tool. The project's research
question, scientific design, methodological choices, experimental protocols, and
interpretation of results were determined by the author. The author reviewed, ran, and
tested the AI-assisted implementation, decided which changes and results to accept or
reject, and can explain the functionality and reasoning behind every submitted
component.

Collaboration with supervisors and Aziz informed the work, but the author remained
responsible for the project decisions, validation, and conclusions. Despite the
assistance described above, the submitted work is the author's own. The git history
records the iterative development and evaluation process.
