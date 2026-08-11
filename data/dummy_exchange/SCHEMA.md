# Blind-exchange detection JSON — schema v1.1

Companion to the C5 blind-exchange protocol (see `fixes-2026-07-29.md`). One
`<blind_id>_detection.json` per `<blind_id>.sgy`.

Changes from v1.0, agreed with Aziz (`colleague-notes-2026-08-09.md`) after the
dummy exchange found the v1.0 format round-tripped but left two things
ambiguous:

- `predicted_time_ms` added — v1.0's `localisation_mask` is 2D (il, xl) only,
  so a detection had no vertical extent and spanned the patch's full sample
  range. That makes IoU degenerate (caps near 6% regardless of detection
  quality, since a true anomaly is ~30 of 462 samples tall). This field gives
  a point estimate of where in time the detection sits.
- Units on every time field are now stated explicitly below. The v1.0 dummy
  exchange's one real ambiguity was `time_ms` vs. sample index — silent on
  the wire, would have produced a plausible wrong answer rather than an
  error.

## Fields

| field | type | units / convention |
|---|---|---|
| `schema_version` | string | this document's version, `"1.1"` |
| `detector_side` | string | whose detector produced this file |
| `blind_id` | string | matches the corresponding `.sgy` filename |
| `patch_origin.il` | int | inline index of the patch's lower corner, **trace index**, not an offset |
| `patch_origin.xl` | int | crossline index of the patch's lower corner, **trace index**, not an offset |
| `patch_origin.time_ms` | float | time of the patch's first sample, **milliseconds from survey t=0** — NOT a sample index |
| `patch_dimensions.n_il` / `n_xl` | int | patch extent in traces |
| `patch_dimensions.n_samples` | int | patch extent in time samples |
| `patch_dimensions.dt_ms` | float | sample interval, milliseconds |
| `is_dhi` | bool | detector's positive/negative call |
| `predicted_tier` | string | one of the severity tier names |
| `confidence` | float | detector's own score, `[0, 1]` |
| `predicted_time_ms` | float | **new in v1.1.** Time of the detection's peak/centroid, **milliseconds from `patch_origin.time_ms`** (i.e. add to `patch_origin.time_ms` to get absolute survey time). Enables real IoU instead of the degenerate full-trace-height case. |
| `localisation_mask.axis_convention` | string, literal | must be exactly `"map_view (il, xl), indices local to this patch (0,0) = (il_origin, xl_origin)"` — parsers should raise on anything else, not attempt to interpret prose |
| `localisation_mask.shape` | `[int, int]` | `[n_il, n_xl]`, matches `patch_dimensions` |
| `localisation_mask.mask` | 2D int array (0/1) | detection footprint in map view only — no vertical/time information, see `predicted_time_ms` for that |

## Still open

`predicted_time_ms` (above) is implemented in v1.1 and is a single point
estimate, not a range. A possible future upgrade — NOT part of v1.1, not
implemented — would replace it with a `t_min_ms`/`t_max_ms` pair for more
precise IoU scoring. Revisit only if the point estimate proves too coarse
once real (non-dummy) exchanges are scored.
