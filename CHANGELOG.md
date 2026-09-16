# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions: SemVer.

## [Unreleased]

### Added

- Packaging for PyPI: project URLs, classifiers, keywords; `CITATION.cff`.
- CI (`ci.yml`): tests on Python 3.10–3.13, ruff lint + format check, build +
  `twine check`. The repo had no `.github/workflows` at all despite 60 tests
  (independent review panel, 2026-09-15).
- Tag-driven release workflow (`release.yml`) using PyPI Trusted Publishing.

## [0.3.1] - 2026-09-02

Licence (GPL-3.0-or-later) and provenance wording. See git history for the
extract-and-leave-source-alone origin of this package.
