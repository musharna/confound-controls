# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions: SemVer.

## [Unreleased]

## [0.4.0] - 2026-09-18

### Fixed

- **A complete, selective match that balanced nothing was scored as a valid
  control.** With a pool larger than the positive set, greedy 1:1 matching runs
  out of nearby negatives and pairs the remaining positives with distant ones.
  On 300 positives and 900 negatives with a score that was only GC, matched on
  GC, 0.3.2 returned `confound-robust` (AUROC 0.807, battery PASS).
  `evaluate_confound` now computes the standardized mean difference of every
  matched column and raises above `max_smd` (default 0.1). **Behaviour
  change:** a battery that ran before can now raise; pass a `caliper`, or
  `max_smd=None` to get the old scoring with the imbalance reported.
- An incomplete match was scored with ALL positives against the fewer matched
  negatives, putting the unmatched positives back into a comparison reported as
  1:1. Only matched positives are scored now; `n_positives` is that count.

### Added

- `caliper=` on `match_negatives`, `evaluate_confound` and `run_battery`:
  positives with no unused negative within that many standard deviations are
  left unmatched and reported. With a caliper set, `require_complete_match`
  defaults to False.
- `method="propensity"`: match on the logit of an unpenalised logistic
  propensity score, highest first. Checked against R's MatchIt 4.5.5 on the
  `lalonde` data (`tests/data/`): logits agree to 1e-7, the 184 pairs are
  MatchIt's up to exact ties, post-match SMDs agree to six decimals. The fit is
  a Newton iteration in numpy, and raises when the confounds separate the
  classes instead of returning a diverged score.
- `standardized_mean_differences`, `balance_report` / `BalanceReport` (SMD before
  and after against one yardstick, variance ratios, `require_balanced`),
  `propensity_logit`. `ConfoundResult.balance`; `MatchResult.matched_positions`,
  `.distances`, `.caliper`. `format_battery` prints `[IMBALANCED: max |SMD| = …]`
  and how many positives an incomplete match left out.

## [0.3.2] - 2026-09-16

First PyPI release.

### Added

- Packaging for PyPI: project URLs, classifiers, keywords; `CITATION.cff`.
- CI (`ci.yml`): tests on Python 3.10–3.13, ruff lint + format check, build +
  `twine check`. The repo had no `.github/workflows` at all despite 60 tests
  (independent review panel, 2026-09-15).
- Tag-driven release workflow (`release.yml`) using PyPI Trusted Publishing.

## [0.3.1] - 2026-09-02

Licence (GPL-3.0-or-later) and provenance wording. See git history for the
extract-and-leave-source-alone origin of this package.
