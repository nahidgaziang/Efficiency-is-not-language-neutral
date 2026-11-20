# Changelog

## 1.3.0 (2026-11-14)

Includes all changes listed under 1.2.0, which was not released separately.

### Added
- Fourth round of analyses (stage 15 of the README):
  - `CODE/r4_common.py`: the TurboQuant-style quantizer with the Lloyd-Max iteration run to convergence above 5 bits (codes of up to 5 bits are unchanged);
  - `CODE/r4_tq_gate.py`: distortion check of the quantizer at 1 to 8 bits;
  - `CODE/r4_grid_belebele.py`, `CODE/r4_p1_price.py`: the price of language neutrality on extended budget grids, as an interval;
  - `CODE/r4_p23.py`, `CODE/r4_p5_units.py`, `CODE/r4_p5.py`: plug-in and certified per-language budgets on Belebele, MIRACL and the haystacks;
  - `CODE/r4_p4.py`: one pooled Belebele index with per-language budgets;
  - `CODE/r4_dakshina_download.py`, `CODE/r4_data.py`, `CODE/r4_exclude.py`: Dakshina transfer pairs and their screening against every test bed;
  - `CODE/r4_m1a_embed.py`, `CODE/r4_m1a_eval.py`, `CODE/r4_m1a_analysis.py`, `CODE/r4_m1a_haystack.py`: the transfer probe on Belebele and the haystacks.
  - Outputs go to `RESULTS/r4/` and `DATA/r4/`.
- `paper_figures/fig_cert.py` and `paper_figures/tables_r4.py`: the figure of certified per-language budgets and the detail tables of the online supplement.
- Ledger entries for the fourth round (`paper_numbers/ledger_r3.py`, function `r4`).

### Changed
- `paper_figures/fig_price.py` draws the price on the extended grids, with exact values and intervals.
- Figure heights of the other figure scripts fit the revised layout; fonts and data are unchanged.
- `paper_figures/tables.py`: the layers table names the extended grids.
- Two run labels written by `CODE/rv4_allocation.py` and `CODE/rv6_effect_size.py` now name only the section of the analysis plan.
- `requirements.txt` adds `regex` (used by `CODE/r4_exclude.py`).

## 1.2.0 (not released separately; included in 1.3.0)

### Added
- Price of language neutrality (stage 14 of the README):
  - `CODE/poln_export_belebele.py` rebuilds the Belebele layers of each model and checks them against the stored per-query results;
  - `CODE/poln_merge.py` merges the per-model exports;
  - `CODE/poln_tests.py` runs the registered tests H0 to H6.
  - Outputs go to `RESULTS/poln/`.
- `paper_figures/fig_price.py`: the figure of the price of language neutrality.
- New ledger entries: the allocated price, the price in the cell without censoring, and the growth decision per layer family.

### Changed
- Repository layout:
  - the scripts that compute the paper's numbers moved from `MANUSCRIPT/analysis/` to `paper_numbers/`;
  - the table and figure scripts moved from `MANUSCRIPT/paper/figure_scripts/` to `paper_figures/`;
  - all their output goes to `paper_output/`.
- The environment variable that sets the thread count is now `AUDIT_THREADS` (was `MISSION_THREADS`), and the Kaggle test variables are now `AUDIT_KAGGLE_*`.
- The JSON key written with the thread count is `audit_threads`, and the Kaggle input datasets are named `audit-*`.
- Behaviour is unchanged.
- Table and figure labels of the revised manuscript:
  - figure text is at least 7 pt;
  - the reader study is labelled "original study";
  - the dimension count is written "1,024".

## 1.1.0 (2026-10-27)

### Added
- Analyses of the revised study:
  - the loss mechanism and possible repairs: standardized margins, per-language codebooks, larger PQ budgets, mixed-language indexes, byte allocation, IVF-PQ;
  - the scale test for a second model;
  - two commercial embedding APIs and five vector databases at their documented settings;
  - calibration with predicted languages;
  - a second RAG reader with three-passage contexts;
  - the validation of MIRAGE-Bench's scoring of generated answers.
- Scripts that compute every number, table and figure of the paper (in v1.1.0 under `MANUSCRIPT/`).
- MIT license.

## 1.0.0 (2026-10-17)

### Added
- Code for reproducing the original study:
  - data preparation and embedding;
  - compression and approximate search on Belebele, MIRACL and the Wikipedia haystacks;
  - null models, the scale test, the downstream RAG reader, and per-language calibration (LCFC).
