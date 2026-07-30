# confound-controls

Did your classifier learn the signal, or a confound?

For each confound, restrict the negatives to a subset matched 1:1 to the
positives on that confound, then recompute AUROC. If the signal survives
matching, it is not that confound. If it collapses to chance, it was.

```python
from confound_controls import run_battery, format_battery, bootstrap_auroc

anchor = bootstrap_auroc(df["label"], probs).point   # uncontrolled baseline

results = run_battery(
    df, prob_map,
    spec={"gc": ["gc"], "expression": ["log_expr"], "joint": ["gc", "log_expr"]},
    anchor=anchor,
)
print(format_battery(results, anchor))
```

Nothing is read from disk and nothing is written to it. The frame, the
probabilities, the confound spec and the anchor are all arguments.

## The failure this exists to prevent

**1:1 matching without replacement from a negative pool the same size as the
positive set consumes every negative, whatever the confound values are.** The
matched comparison is then identical to the unmatched one — and a vacuous
control does not announce itself as inconclusive. It returns the original
result, which reads as the confound having been ruled out.

This is not hypothetical. It is how the first version of this library's own
test suite behaved: 150 positives against 150 negatives, a score computed from
nothing but GC, matched on GC, verdict `confound-robust`, AUROC 0.9312.

So `evaluate_confound` refuses by default:

```
ValueError: matching consumed the entire negative pool (150 matched from 150
available), so it selected nothing and the control is vacuous -- the matched
comparison equals the unmatched one and will report the confound as ruled out.
```

Pass `require_selective_match=False` if you mean it; the result then carries
`match_selective=False` and `format_battery` prints `[VACUOUS CONTROL]`.

## Verdicts

| verdict           | meaning                                                                             |
| ----------------- | ----------------------------------------------------------------------------------- |
| `confound-robust` | enough of the anchor's above-chance signal survived, and the interval clears chance |
| `confound-driven` | the interval straddles chance — matching removed the signal                         |
| `partial`         | signal is real but diminished                                                       |
| `inconclusive`    | the interval is too wide to support any of the above (opt in via `max_ci_width`)    |

`inconclusive` has no counterpart in the source this came from, which folded
those cases into `confound-driven` — reporting an absent measurement as a
finding.

## Two more controls

**Incremental validity** — does the new feature add anything *beyond* the
confounds? Fit confound-only and confound+feature models, evaluate both on the
same held-out rows, and put a **paired** bootstrap interval on the AUROC
difference.

Paired is the point. Two independent intervals on two AUROCs overlap far more
often than the interval on their difference excludes zero, because the pair is
computed on the same resampled rows and the shared variance cancels. Comparing
two separately-reported AUROCs by eye is exactly the error this prevents.

```python
from confound_controls import incremental_validity
res = incremental_validity(conf_train, conf_test, feat_train, feat_test, y_train, y_test)
res.verdict   # adds-signal | harms | no-added-signal | underpowered
```

`feat_train` must be out-of-fold. Fitting the feature on the rows the confound
model is evaluated against leaks the label into the augmented model and
manufactures the very lift the control is testing for.

The source's verdict was two-valued (`"FM-adds-signal" if lo > 0 else
"no-added-signal"`), so an interval straddling zero, an interval too wide to
say anything, and an interval lying entirely *below* zero all reported the
same. The last of those is an augmented model that is reliably **worse** — a
finding, not an absence of one.

**Ablation** — destroy the structure of interest, keep the nuisance properties,
score with the same model, and see what survives.

```python
from confound_controls import assert_ablation_changed_input, ablation_control
assert_ablation_changed_input(real_inputs, ablated_inputs)   # do this first
res = ablation_control(y, p_real, p_ablated)
res.verdict   # structure-dependent | structure-independent | partial | inconclusive
```

That first call is not optional politeness. **A broken ablation leaves the
input unchanged, so the scores match exactly, the delta is zero, and the
control reports the model as surviving** — the strongest possible result,
obtained by doing nothing. No score-level statistic can catch it, which is why
the check belongs on the input. The source had this instinct
(`assert not np.allclose(shuffled, real)`); here it is a first-class function
that also handles non-float inputs like sequences, where `np.allclose` raises.

This is the same shape as the vacuous matched control above: an inert control
returns the original answer, which reads as the hypothesis surviving.

## What changed from the source

Extracted from `arf_promoter_analysis/scripts/aim3_matched_negatives.py`,
present byte-identically in `phelipanche-fm`. The code had been copied between
two independent repositories, which is what marked it as worth extracting:
duplication across repos is revealed reuse demand rather than a proxy for it.

The matching and the statistics are unchanged. What changed:

- **`ANCHOR = 0.629` was a module-level constant** — one study's own baseline
  AUROC compiled into the library, against which every imported use silently
  scored. Now a required argument, because there is no defensible default for
  "what does good look like in your problem".
- **Input paths were derived from `__file__`** and the confound columns
  (`gc`, `log_basemean`, `og_size`, `n_frac`, 64 `kmer3_*`) were hardcoded, so
  the module ran against one dataset in one repo layout. Both are arguments.
- **Incomplete matches were silent.** When the pool ran out the original
  returned fewer negatives than positives and said nothing — and the drop is
  not random, it hits the positives in the densest region of confound space.
  `MatchResult.complete` now answers this, and it is enforced by default.
- **The pass flag counted a hardcoded subset** of the study's own confound
  names, so a confound added to the spec was scored, printed, and then left out
  of the decision it existed to inform. `battery_passes` counts every result.
- **`recovery()` divided by `anchor - 0.5` without checking it.** An anchor at
  or below chance makes that ratio a non-quantity rather than a large number;
  it is now refused.
- **The bootstrap silently skipped single-class resamples.** With few positives
  most resamples are skipped and the interval comes from a handful of
  replicates. It now raises rather than returning a confident-looking number.

## Not extracted

`aim3_confound_table.py` builds the confound table itself from FASTA, DE
results and orthogroup sizes. That is study data-prep rather than a control,
and its sequence-composition code would duplicate what `seqbench` already
does — so it stays where it is.

## The source repos are untouched

Both `arf_promoter_analysis` and `phelipanche-fm` are active. This is
extract-and-leave-source-alone: nothing was refactored there, and neither repo
imports this package.

## Install

```bash
pip install -e ".[test]"
pytest -q
```

Requires numpy, pandas, scipy, scikit-learn.
