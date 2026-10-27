# Efficiency Is Not Language-Neutral

Code for the study *Efficiency Is Not Language-Neutral: Auditing and Calibrating Compression, Approximate Search, and
Similarity Thresholds in Multilingual Dense Retrieval*.

Author: Nahid Gazi ([ORCID 0009-0006-1661-9985](https://orcid.org/0009-0006-1661-9985)), Department of Computer Science
and Engineering, Shahjalal University of Science and Technology, Sylhet, Bangladesh

Archived releases: all versions [10.5281/zenodo.23434744](https://doi.org/10.5281/zenodo.23434744); v1.0.0
[10.5281/zenodo.23434745](https://doi.org/10.5281/zenodo.23434745).

The code measures how vector compression (product, scalar, binary, rotation-based and residual quantization; PCA and
Matryoshka truncation), approximate nearest-neighbor search (HNSW, IVF) and global similarity thresholds distribute
their retrieval loss across languages. It covers eight embedding models on Belebele (122 language variants), MIRACL (18
languages), Wikipedia haystacks (40 languages and 4 romanized query groups) and a scale test with up to 4.98 million
passages, followed by a downstream RAG reader and a per-language calibration with finite-sample guarantees (LCFC).

Version 1.1.0 adds the analyses of the revised study: re-analyses of the loss mechanism and of possible repairs
(standardized margins, per-language codebooks, larger PQ budgets, mixed-language indexes, byte allocation, IVF-PQ), the
scale test for a second model, two commercial embedding APIs, five vector databases at their documented settings,
calibration with predicted languages (GlotLID), a second RAG reader with three-passage contexts, the validation of
MIRAGE-Bench's scoring for generated answers, and the scripts that compute every number, table and figure of the paper.

## Repository layout

| Path | Content |
|---|---|
| `CODE/` | data preparation, embedding, evaluation and analysis scripts |
| `CODE/tools/` | input export, file validation, independent checks |
| `KAGGLE/` | scripts that ran on Kaggle (2 x NVIDIA T4) |
| `MANUSCRIPT/analysis/` | the results ledger: every number of the paper, computed from `RESULTS/` |
| `MANUSCRIPT/paper/figure_scripts/` | tables and figures of the paper |
| `EXPERIMENTS/diagnostics/` | two diagnostics of the generated-answer study |
| `DATA/` | created at run time: datasets, embeddings, reference files (not in the repository) |
| `RESULTS/` | created at run time: all outputs (not in the repository) |

Scripts locate `DATA/` and `RESULTS/` relative to the repository root. Most scripts refuse to overwrite an existing
output folder.

## Environment

- Python 3.12.10; packages in `requirements.txt` (PyTorch 2.6.0 with CUDA 12.4 for the embedding steps; matplotlib for
  the figures).
- Embeddings: NVIDIA RTX 3050 (6 GB), fp16, except where stated below. Analyses: CPU (Intel i5-13420H, 15.6 GB RAM).
- Kaggle (2 x T4, 16 GB each): Qwen3-Embedding-4B encoding, the scale tests (encoding and the pooled Task B indexes)
  and the studies of stage 13. Each Kaggle script installs the packages it needs at run time; the versions are pinned
  in the scripts and written to each run's log.
- `EXPERIMENTS/diagnostics/d6_gate_20261024/d6_gate_diag_summary.py` also needs `mirage-bench==0.0.4` and
  `langdetect==1.0.9`. mirage-bench depends on vLLM, which has no Windows build; on Windows, install mirage-bench's
  other dependencies and provide `vllm` as an import-only stub (the diagnostic does not call it).
- Figures use the Linux Biolinum O font. Copy `LinBiolinum_R.otf`, `LinBiolinum_RB.otf` and `LinBiolinum_RI.otf`
  (SIL Open Font License; part of the `libertine` TeX package, e.g. `kpsewhich LinBiolinum_R.otf`) into
  `MANUSCRIPT/paper/figure_scripts/fonts/`.

```
pip install -r requirements.txt
```

## Determinism

- `CODE/common.py` sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` from
  `MISSION_THREADS` only if they are unset, and only when it is imported before numpy. Run every script in a fresh
  process with these four variables unset and `MISSION_THREADS` set. `CODE/tools/run_measured.py` does this and also
  records wall time and peak memory:
  ```
  python CODE/tools/run_measured.py <log file> -- python CODE/<script>.py <args>
  ```
  `run_measured.py` runs the command with `CODE/` as the working directory, so directory arguments should be absolute
  paths.
- PQ, OPQ, PCA and k-means fits depend on the BLAS thread count. All reported results used `MISSION_THREADS=4`, except
  the MIRACL steps of multilingual-e5-small, which used 12 threads: `miracl_eval.py e5s`, the LCFC MIRACL first stages
  (`lcfc_tables.py fit miracl e5s <stage>`), `rv_miracl.py e5s`, `rv5_ivfpq.py e5s`, `r3_miracl.py e5s` and
  `r3_d6_prep.py e5s --part e2|alloc`. `rv_belebele.py`, `rv_haystack.py`, `rv_miracl.py`, `rv5_belebele_pq.py`,
  `rv5_ivfpq.py`, `r3_gaps.py`, `r3_ers.py`, `r3_haystack.py`, `r3_miracl.py`, `r3_d6_prep.py`, `r3_d7_prep.py` and
  `a1_eval.py` stop unless `MISSION_THREADS` has the value used for the reported results.
- The evaluation scripts of stages 11 and 12 first check the layers they share with earlier stages against the stored
  per-query results (maximum difference 1e-9) and stop on a mismatch.
- HNSW graphs built with several threads are not bit-identical between builds; HNSW results are reproducible in
  distribution, not bit for bit.
- Random seeds are fixed in the scripts (bootstrap seeds 20261006, 20261007 and 20261015; fit seeds 0, 1, 2;
  byte-allocation splits, answer generation and judge 20261019; judge rerun 20261020).

## Data

| Data | Source | Prepared by |
|---|---|---|
| Belebele, test split, 122 language variants | `facebook/belebele`, revision `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` | `CODE/embed_belebele.py` (loaded with `datasets`) |
| MIRACL dev, 18 languages | `mteb/MIRACLRetrievalHardNegatives`, revision `332a9acb49f5e83d5397683f79d23e588f685916` | `CODE/get_miracl.py`; 40,000 passages per language drawn by `CODE/embed_miracl.py` (seed 20261006) |
| Wikipedia haystacks | `wikimedia/wikipedia`, 20231101 dumps | `CODE/build_haystack.py` (20,000 paragraphs per language plus the Belebele gold passages) |
| Scale test | `miracl/miracl-corpus` v1.0, up to 300,000 passages per language | `CODE/tools/export_scale_inputs.py`, then `KAGGLE/scale_miracl_bgem3.py` |
| MIRAGE-Bench dev.small, 18 languages | `nthakur/mirage-bench`, revision `148cc85b0ef10b2ec57d28d06391eaa75b7891c9`, files `<lang>/dev.small-00000-of-00001.parquet` | saved unchanged as `DATA/external/mirage_bench_dev_small/<lang>_dev.small.parquet` |
| MIRAGE-Bench released outputs | `nthakur/mirage-bench-output`, revision `025145142be936d525ef7f655c32dc32b08ea0e5`, files `<lang>/dev-00000-of-00001.parquet` | saved unchanged under `DATA/external/mirage_bench_output_025145142be/<lang>/`; `KAGGLE/d6_score.py` downloads its own copy |
| Resource classes | see the table below | saved unchanged in `DATA/external/r3_5_resource_classes/`; `CODE/tools/joshi_classes.py` writes `belebele_classes.csv` there |
| MTEB reference results | `embeddings-benchmark/results` on GitHub | downloaded by `CODE/tools/a1_mteb_check.py` |

| Resource-class file | URL | SHA-256 |
|---|---|---|
| `lang2tax.txt` | https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt (Joshi et al., ACL 2020) | `2E182E8F4BD9ADB0BC04AF2F74852D770581AC0B5A3609224CED7B08266D6DD2` |
| `belebele_README.md` | https://raw.githubusercontent.com/facebookresearch/belebele/main/README.md | `F7A9AEAD9AF9056C96837EEB8EAFA301EC37A4512E52520F1C3D7903B69CE145` |
| `iso-639-3.tab` | https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3.tab | `7A5AC370703EA0897356A8C383F2147865CAF479E41C1C7BD96CC4C5F65E6907` |
| `iso-639-3-macrolanguages.tab` | https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3-macrolanguages.tab | `FB01A86376D9C1ABFC96D16BE1B6DCFFB4776A1F52FA22338D4979E9FFE1822F` |

`lang2tax.txt` maps language names, not codes, to the classes 0-5 of Joshi et al. `joshi_classes.py` matches each
Belebele variant in this order: the ISO 639-3 reference name of its code; Belebele's English name without
"(Romanized)"; the reference name of the code's ISO macrolanguage. Names with more than one class and unmatched variants
are excluded; romanized variants inherit the class of their language.

Each dataset and model keeps its own license; see its card on Hugging Face.

## Models

| Tag | Model | Revision |
|---|---|---|
| `e5s` | intfloat/multilingual-e5-small | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `minilm` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| `bgem3` | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` |
| `qwen3e_ep` | Qwen/Qwen3-Embedding-0.6B, developer evaluation prompts | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `qwen3e_ep_nt` | the same model and prompts, passages not truncated | as above |
| `qwen3e4b_ep` | Qwen/Qwen3-Embedding-4B, developer evaluation prompts (Kaggle) | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| `arcticl2` | Snowflake/snowflake-arctic-embed-l-v2.0 | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` |
| `egemma` | google/embeddinggemma-300m (bfloat16) | `57c266a740f537b4dc058e1b0cda161fd15afa75` |
| `jinav5` | jinaai/jina-embeddings-v5-text-small-retrieval (CC BY-NC 4.0) | `6856e76bb72982e58de0620458a4e8b3614da340` |
| `qwen3e`, `qwen3e4b` | the two Qwen3 models with an earlier query prompt; used only for the prompt comparison | as above |
| `l3bn` | l3cube-pune/bengali-sentence-similarity-sbert (Bangla case study) | `3af011cee9743eee4034904a97a47a769258d5b8` |
| `voyage` | voyage-4-large (Voyage AI API) | API |
| `openai` | text-embedding-3-large (OpenAI API) | API |
| RAG reader | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| second RAG reader | meta-llama/Llama-3.2-3B-Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| generation readers | Qwen/Qwen2.5-7B-Instruct; meta-llama/Llama-3.1-8B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28`; `0e9e39f249a16976918f6564b8830bc894c89659` |
| generation judge | meta-llama/Meta-Llama-3-8B-Instruct | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| language identification | cis-lmu/glotlid, `model_v3.bin` | `85cd6716494360367b75f642b5bc78667605d0b4` |

Query and passage prompts are defined in `CODE/common.py` (`MODELS`, `QUERY_PROMPTS`), `CODE/embed_belebele.py` and
`CODE/embed_belebele_new.py`. `python CODE/tools/prefetch_models.py <repo id> ...` downloads a model into the project
cache and checks every file size. The Kaggle scripts of stage 13 load the current revision of each model they
download; the revisions listed for the second RAG reader, the generation readers, the judge and GlotLID are those
recorded by the reported runs.

## Analysis names

Output folders and run records use these names.

| Name | Analysis |
|---|---|
| E1 | Belebele compression sweep: mean and worst-language retention per layer and budget |
| E2, Task A′ / B′ | MIRACL: per-language 40k indexes / pooled 720k index (HNSW, IVF) |
| H-A / H-B | haystacks: per-language 20,488-passage indexes / pooled 818k index |
| M7 | null-model decomposition (matched-mean noise, residual-norm null) |
| E3a, R-H, R-G | label-free audits (agreement with exact search; full-precision score gap) |
| F5 | text length and truncation control |
| M5 | PCA fitting population |
| E5, E5b, E5b-H, E5c | similarity thresholds (global, per language, CSLS, ratio margin; language-identification errors) |
| R-A to R-F | robustness: absolute loss, floors and leave-worst-out, seed-aware intervals, multiplicity, cross-model profile, corpus size, re-ranking |
| WP1 | compression premium |
| WP2 (T1, T2, T4) | RaBitQ, TurboQuant-style, residual and fast-scan quantizers; compounding, null model, oversampled rescoring |
| WP3 (T3) | share of a language in the quantizer's training sample |
| WP4 (T5) | compression-trained models |
| S, Task A-scale / B-scale | scale test |
| E6 | RAG reader: per-language answer loss |
| LCFC (J1-J4) | per-language calibration of re-rank depth, IVF nprobe and HNSW efSearch |
| C-1 to C-10 | additional checks: order-statistics null, passage-cluster bootstrap, grouped calibration splits, robust forms, independent recomputation, re-runs |
| S1 | Qwen3-Embedding-0.6B with untruncated passages |
| S2 | scale test with multilingual-e5-small; comparison with BGE-M3 |
| S7 | variance decomposition of the loss |
| RV1 | standardized score margin as an account of the per-language loss |
| RV2 | per-language PQ codebooks (arm L: fit on the language; arm L-eq: fit on as many passages as the pooled sample holds) |
| RV3 | mixed-language index |
| RV4 | language-aware byte allocation at equal memory, with and without labels |
| RV5-a / b / c | PQ at d/8 and d/4 bytes / exact re-ranking depth / IVF-PQ with exact refine on the pooled MIRACL index |
| RV6 | effect-size re-analyses |
| R3-1 | regularity of worst-language against mean retention |
| R3-2 | equivalence across index sizes in the scale test |
| R3-3 | one loss curve across models and testbeds |
| R3-4 | distractor dose on the haystacks (mediation by the margin) |
| R3-5 | loss by resource class (descriptive) |
| R3-6 | LCFC over byte budgets |
| R3-7 (b) / (c) | exact maximin allocation against the greedy / allocation on MIRACL |
| R3-8 | label-free forecast of the worst languages from exact-search gaps |
| A1 | commercial embedding APIs and their compressed formats |
| A2 | vector databases at documented settings |
| A3 | statistics of the equity gap against a normal model |
| B4 | parameter-free loss formula |
| C5 | equitable residual scaling (ERS): per-language standardization with one shared PQ16 codebook |
| S8 | per-language calibration with languages identified by GlotLID |
| D6 | generated answers scored with MIRAGE-Bench, behind a validation gate |
| D7 | second reader family and three-passage contexts for E6 |
| J-a to J-e | inputs computed for the items above: haystack codebook arms and distractor dose, MIRACL top-1 fidelity, exact top-10 margins and top-2 gaps |

Labels such as `EXPERIMENT_PLAN §26 A2` in output files and in the ledger's `source` column name the section of the
time-stamped analysis plan under which an analysis was registered (see the paper's section on the analysis plan). The
plan itself is not part of this repository.

## Reproduction

Commands are run from the repository root with `MISSION_THREADS=4` unless stated (the `MISSION_THREADS=12` prefix
below is POSIX shell syntax; on Windows set the variable before the command). Steps within a stage can run in any order
once their inputs exist; every step reads only files written by earlier steps.

### 1. Data and embeddings

```
python CODE/get_miracl.py
python CODE/build_haystack.py

python CODE/embed_belebele.py e5s belebele
python CODE/embed_belebele.py minilm belebele
python CODE/embed_belebele.py bgem3 belebele
python CODE/embed_belebele.py qwen3e belebele

python CODE/embed_miracl.py e5s
python CODE/embed_miracl.py bgem3
python CODE/embed_miracl.py minilm
python CODE/embed_miracl.py qwen3e
python CODE/embed_miracl.py arcticl2
python CODE/tools/check_miracl_npz.py <tag>

python CODE/embed_haystack.py e5s
python CODE/embed_haystack.py minilm
python CODE/embed_haystack.py bgem3
python CODE/embed_haystack.py qwen3e
python CODE/embed_haystack.py l3bn --langs ben_Beng
python CODE/tools/check_haystack_npz.py <tag>

python CODE/embed_queries_reprompt.py qwen3e qwen3e_ep
```

Qwen3-Embedding-4B (Kaggle):

```
python CODE/tools/export_kaggle_inputs.py
```
Upload `KAGGLE/inputs/` as a Kaggle dataset and run `KAGGLE/qwen3e4b_belebele.py` on 2 x T4. Then:
```
python CODE/tools/extract_kaggle_result.py <result zip> qwen3e4b
python CODE/tools/check_kaggle_npz.py qwen3e4b
```
Run `KAGGLE/qwen3e4b_queries_ep.py` on Kaggle for the corrected query prompt, place its `belebele_queries.npz` and
`belebele_queries_info.json` in `DATA/emb/qwen3e4b_ep_kaggle/`, and run `python CODE/tools/integrate_kaggle_queries.py`.

Compression-trained models:

```
python CODE/embed_belebele_new.py arcticl2
python CODE/embed_belebele_new.py egemma
python CODE/embed_belebele_new.py jinav5 --smoke
python CODE/tools/check_jinav5.py k1 <smoke log>
python CODE/tools/check_jinav5.py k2
python CODE/embed_belebele_new.py jinav5
python CODE/tools/check_jinav5.py k3
python CODE/tools/fix_npz_ids.py arcticl2
python CODE/tools/fix_npz_ids.py egemma
python CODE/tools/fix_npz_ids.py jinav5
python CODE/tools/fetch_mteb_wp4.py
python CODE/tools/config_audit_wp4.py arcticl2 --miracl
python CODE/tools/config_audit_wp4.py egemma
python CODE/tools/config_audit_wp4.py jinav5
```

Configuration audit of the other models against published MTEB results: `python CODE/tools/config_audit.py` and
`python CODE/tools/config_audit_qwen3e4b_ep.py`.

### 2. Belebele: compression sweep, null model, controls

```
python CODE/belebele_eval.py e5s --seeds 3 --opq-seeds 2
python CODE/belebele_eval.py minilm --seeds 3 --opq-seeds 2
python CODE/belebele_eval.py bgem3 --seeds 2 --opq-seeds 1
python CODE/belebele_eval.py bgem3 --seed-list 2 --opq-seed-list 1 --out-tag bgem3_d5
python CODE/belebele_eval.py qwen3e_ep --seeds 2 --opq-seeds 1
python CODE/belebele_eval.py qwen3e4b_ep --seeds 2 --opq-seeds 1
python CODE/belebele_eval.py arcticl2 --seeds 3 --opq-seeds 2 --opq-budgets 16,32 --pca-dims 128,256,512
python CODE/belebele_eval.py egemma --seeds 3 --opq-seeds 2 --opq-budgets 16,32 --pca-dims 96,192,384
python CODE/belebele_eval.py jinav5 --seeds 3 --opq-seeds 2 --opq-budgets 16,32 --pca-dims 128,256,512
python CODE/stats_belebele_ci.py <tag>
python CODE/tools/independent_check.py <tag>
python CODE/tools/build_allseeds_perlang.py
python CODE/t13_allseeds.py

python CODE/null_decomposition.py <tag>
python CODE/null_decomposition.py bgem3 --fit-seed 1 --cells pq16,pq32,opq16 --suffix _seed1
python CODE/pca_population.py <tag>
python CODE/tokenization_stats.py e5s minilm bgem3
python CODE/tools/token_stats_local.py qwen3e_ep
python CODE/tools/token_stats_local.py qwen3e4b_ep
python CODE/f5_analysis.py <tag>
python CODE/rerank_eval.py <tag>
python CODE/rerank_eval.py qwen3e4b_ep --no-opq
python CODE/robustness_checks.py --out v4_20261015_5models --m7-tags e5s,minilm,bgem3,qwen3e_ep,qwen3e4b_ep --e1-tags e5s,minilm,bgem3,qwen3e_ep,qwen3e4b_ep --re-tags e5s,minilm,bgem3,qwen3e_ep
python CODE/gap_audit.py --tags e5s,minilm,bgem3,qwen3e_ep,qwen3e4b_ep --out v3_20261015_5models
```

`<tag>` takes these values:
- `stats_belebele_ci.py`, `independent_check.py`, `null_decomposition.py`: `e5s minilm bgem3 qwen3e_ep qwen3e4b_ep
  arcticl2 egemma jinav5`;
- `pca_population.py`: `e5s minilm bgem3 qwen3e4b_ep`;
- `f5_analysis.py`: `e5s minilm bgem3 qwen3e_ep qwen3e4b_ep`;
- `rerank_eval.py`: `e5s minilm bgem3 qwen3e_ep`.

### 3. MIRACL

```
MISSION_THREADS=12 python CODE/miracl_eval.py e5s
python CODE/miracl_eval.py e5s --pooled-only
python CODE/miracl_eval.py e5s --seed 1
python CODE/tools/compare_seeds.py e5s
python CODE/miracl_eval.py bgem3 --pooled
python CODE/miracl_eval.py minilm --pooled
python CODE/miracl_eval.py qwen3e_ep --pooled
python CODE/miracl_eval.py arcticl2
python CODE/stats_ci.py miracl <tag>
python CODE/stats_ci.py miracl <tag> --pooled
python CODE/miracl_labelfree.py <tag>
python CODE/miracl_thresholds.py <tag>
python CODE/thresholds_v2.py <tag>
python CODE/thresholds_v2.py <tag> --rp pooled
python CODE/thresholds_v2.py <tag> --lid
python CODE/default_configs.py miracl e5s
python CODE/default_configs.py miracl bgem3
python CODE/allocation.py e5s
python CODE/allocation.py e5s --suffix _rerun
python CODE/stats_allocation_ci.py e5s --suffix _rerun
python CODE/allocation.py bgem3
python CODE/ann_mediation.py e5s
python CODE/ann_mediation.py bgem3
python CODE/miracl_adaptive_ann.py e5s
python CODE/diag_outlier.py e5s yo,sw,te,de,en,zh
python CODE/diag_dupes.py
python CODE/diag_dedup.py
python CODE/diag_negative_loss.py
```

`<tag>`: `stats_ci.py` and `miracl_labelfree.py`: `e5s minilm bgem3 qwen3e_ep`; `miracl_thresholds.py` and
`thresholds_v2.py`: `e5s bgem3`.

### 4. Haystacks

```
python CODE/haystack_eval.py <tag>
python CODE/haystack_eval.py l3bn --file haystack_ben_Beng.npz --skip-pooled
python CODE/tools/haystack_summary_B_floor.py <tag>
python CODE/stats_ci.py haystack <tag>
python CODE/haystack_thresholds_v2.py <tag>
python CODE/default_configs.py haystack e5s
python CODE/default_configs.py haystack bgem3
```

`<tag>` is `e5s minilm bgem3 qwen3e_ep`; `haystack_summary_B_floor.py` ran for `e5s minilm bgem3`.

### 5. Modern and default quantizers, training share, compression-trained models, premium

```
python CODE/modern_quant.py <tag>
python CODE/modern_quant_miracl.py <tag> --units tq,rabitq,pqfs
python CODE/modern_quant_miracl.py <tag> --units rq
python CODE/modern_quant_haystack.py <tag> --units tq,rabitq,pqfs
python CODE/modern_quant_haystack.py <tag> --units rq
python CODE/tools/rq_mse_check_miracl.py
python CODE/null_modernq.py <tag>
python CODE/wp2_analysis.py
python CODE/wp2_analysis_miracl.py
python CODE/wp2_analysis_haystack.py
python CODE/wp2_t2.py
python CODE/tools/diag_wp2.py

python CODE/share_intervention.py bgem3
python CODE/share_intervention.py bgem3 --layers opq16
python CODE/share_intervention.py e5s --split --layers pq16,pq32
python CODE/share_intervention.py qwen3e_ep --split --layers pq16,pq32
python CODE/wp3_analysis.py bgem3 --layers pq16,pq32,rabitq1,opq16 --out analysis_all_layers
python CODE/wp3_analysis.py e5s_split --layers pq16,pq32
python CODE/wp3_analysis.py qwen3e_ep_split --layers pq16,pq32

python CODE/modern_quant_miracl.py arcticl2 --units tq,rabitq,pqfs
python CODE/modern_quant_miracl.py arcticl2 --units rq
python CODE/t12_arctic_miracl.py
python CODE/wp4_analysis.py

python CODE/compression_premium.py
python CODE/compression_premium.py --wp2
```

`modern_quant.py`: the eight Belebele models; `modern_quant_miracl.py` and `modern_quant_haystack.py`: `e5s bgem3
qwen3e_ep`; `null_modernq.py`: `e5s minilm bgem3 qwen3e_ep qwen3e4b_ep`. `share_intervention.py bgem3` needs the scale
test output of stage 6.

### 6. Scale test

BGE-M3:
```
python CODE/tools/export_scale_inputs.py bgem3 <repository root>/KAGGLE/scale_inputs
```
Upload `KAGGLE/scale_inputs/` to Kaggle and run `KAGGLE/scale_miracl_bgem3.py`. Check the downloaded output with
`python CODE/tools/check_scale_output.py <download folder>` (the folder that contains `scale_bgem3/`), then copy the
contents of `scale_bgem3/` into `DATA/emb/bgem3/scale/`. Then:
```
python CODE/scale_eval.py
python CODE/scale_analysis.py
python CODE/tools/export_taskB_inputs.py
```
Upload `KAGGLE/inputs_taskB/` and the scale output to Kaggle, run `KAGGLE/scale_taskB.py`, and place its output folder
(with `units/`) at `RESULTS/scale/bgem3/taskB_kaggle/`. Then:
```
python CODE/scale_analysis_taskB.py --out analysis_B
python CODE/scale_s1_robust.py
```

multilingual-e5-small (after the BGE-M3 scale test and stage 3):
```
python CODE/tools/export_scale_e5s_inputs.py
```
Upload `KAGGLE/inputs_scale_e5s/` to Kaggle and run `KAGGLE/scale_miracl_e5s.py`. Check the output with
`python CODE/tools/check_scale_output.py <download folder> --tag e5s`, then copy the contents of `scale_e5s/` into
`DATA/emb/e5s/scale/`. Then:
```
python CODE/scale_eval.py --tag e5s
python CODE/scale_analysis.py --tag e5s
```
Run `KAGGLE/scale_taskB_e5s.py` with `KAGGLE/inputs_scale_e5s/` and the scale output attached, and place its output
folder (with `units/`) at `RESULTS/scale/e5s/taskB_kaggle/`. Then:
```
python CODE/scale_analysis_taskB.py --tag e5s
python CODE/scale_s1_robust.py --tag e5s
python CODE/s2_compare.py
```

### 7. Downstream RAG reader

```
python CODE/rag_reader.py e5s
python CODE/rag_reader.py e5s --rest-of-v1 --suffix _rest
python CODE/rag_reader.py bgem3
python CODE/tools/pooled_top1_lang.py e5s
python CODE/tools/pooled_top1_lang.py bgem3
python CODE/rag_reader.py e5s --pooled
python CODE/rag_reader.py bgem3 --pooled
python CODE/rag_analysis.py --out e6_v2_20261015
python CODE/rag_analysis.py --out e6_pooled_20261016 --pooled
```

### 8. Per-language calibration (LCFC)

Tables, for every combination of dataset `belebele | miracl | haystack`, model `e5s | bgem3 | qwen3e_ep` and first
stage `pq16 | opq16 | pq32 | pca64 | bin_gc`, plus Belebele `qwen3e4b_ep` without `opq16`:
```
MISSION_THREADS=12 python CODE/lcfc_tables.py fit miracl e5s <stage>
python CODE/lcfc_tables.py rerank <dataset> <tag> <stage>
python CODE/lcfc_tables.py pooled <miracl|haystack> <tag> exact
python CODE/lcfc_tables.py pooled <miracl|haystack> <tag> ivf
python CODE/lcfc_tables.py pooled <miracl|haystack> <tag> hnsw
```
Calibration, report and checks:
```
python CODE/lcfc_calibrate.py --test
python CODE/lcfc_calibrate.py run
python CODE/lcfc_calibrate.py summary
python CODE/lcfc_report.py
python CODE/tools/lcfc_shift_summary.py
python CODE/tools/lcfc_indep_check_j4.py
python CODE/tools/lcfc_kmeans_repro.py
python CODE/tools/lcfc_check_rerank_labels.py <dataset> <tag> <stage>
python CODE/tools/lcfc_check_miracl_labels.py <tag>
python CODE/tools/lcfc_check_haystack_labels.py <tag>
python CODE/tools/lcfc_e2_thread_check.py
```

### 9. Additional checks

```
python CODE/assurance_c1_c2.py
python CODE/assurance_c2_e6.py
python CODE/assurance_c3.py
python CODE/assurance_c4b.py
python CODE/assurance_c5.py
python CODE/assurance_c6.py
python CODE/assurance_c9.py
python CODE/tools/c10_question_order_check.py
python CODE/assurance_c10.py
python CODE/tools/c10_t2_thread_check.py
python CODE/tools/recheck_answer_loss_bound.py
```

### Query-prompt comparison (Qwen3)

The comparison of the earlier and the corrected Qwen3 query prompts uses the `qwen3e` embeddings of stage 1:
```
python CODE/tools/qwen3_prompt_check.py qwen3e
python CODE/tools/qwen3_prompt_confirm.py
python CODE/tools/compare_prompts_exact.py qwen3e qwen3e_ep
python CODE/tools/miracl_npz_diag.py qwen3e e5s bgem3
python CODE/tools/reembed_check.py qwen3e
```

### 10. Untruncated passages, variance decomposition, corpus token statistics

`s1_encode_untruncated.py run` re-encodes, on the GPU, the Belebele passages that exceed Qwen3-Embedding-0.6B's
original token limit, so that no passage is truncated, and writes `DATA/emb/qwen3e_ep_nt/`; `smoke <folder>` is a
shorter test run that writes to `<folder>`. It reads `KAGGLE/inputs/belebele_passages.parquet` (stage 1) and the model
from the project cache.
```
python CODE/tools/s1_encode_untruncated.py smoke <folder>
python CODE/tools/s1_encode_untruncated.py run
python CODE/belebele_eval.py qwen3e_ep_nt --seeds 2 --opq-seeds 1
python CODE/tools/token_stats_local.py qwen3e_ep_nt
python CODE/stats_belebele_ci.py qwen3e_ep_nt
python CODE/f5_analysis.py qwen3e_ep_nt
python CODE/s1_analysis.py

python CODE/s7_variance_decomposition.py

python CODE/tools/token_stats_corpora.py miracl e5s minilm bgem3 qwen3e_ep arcticl2
python CODE/tools/token_stats_corpora.py haystack e5s minilm bgem3 qwen3e_ep
```

### 11. Mechanism and repairs (RV1-RV6)

```
python CODE/rv5_belebele_pq.py <tag>
python CODE/rv_belebele.py <tag>
python CODE/rv_belebele.py <tag> --layers <d/8 and d/4 layers>
python CODE/rv_haystack.py <tag>
MISSION_THREADS=12 python CODE/rv_miracl.py e5s
python CODE/rv_miracl.py <tag>
MISSION_THREADS=12 python CODE/rv5_ivfpq.py e5s
python CODE/rv5_ivfpq.py bgem3
python CODE/rv1_analysis.py
python CODE/rv2_analysis.py
python CODE/rv3_analysis.py
python CODE/rv4_allocation.py --variants labeled,labelfree
python CODE/rv5_analysis.py --parts a,b,c
python CODE/rv6_effect_size.py
```

- `rv5_belebele_pq.py` and `rv_belebele.py`: the eight Belebele models. `rv5_belebele_pq.py` writes
  `RESULTS/belebele/<tag>_rv5/`, which `rv_belebele.py --layers` reads. The `--layers` values are `pq_48,pq_96` for
  `e5s` and `minilm`; `pq_96,pq_192` for `egemma`; `pq_128,pq_256` for `bgem3`, `qwen3e_ep`, `arcticl2` and `jinav5`;
  `pq_320,pq_640` for `qwen3e4b_ep`.
- `rv_haystack.py`: `e5s minilm bgem3 qwen3e_ep`; `rv_miracl.py` with 4 threads: `minilm bgem3 qwen3e_ep arcticl2`.

### 12. Mechanism tests on stored embeddings (R3-1 to R3-8, A3, B4, C5)

```
python CODE/r3_gaps.py belebele <tag>
python CODE/r3_gaps.py haystack <tag>
python CODE/r3_ers.py belebele <tag>
python CODE/r3_ers.py haystack <tag>
python CODE/r3_haystack.py <tag> --task arms
python CODE/r3_haystack.py <tag> --task dose
MISSION_THREADS=12 python CODE/r3_miracl.py e5s
python CODE/r3_miracl.py <tag>
python CODE/tools/joshi_classes.py

python CODE/r3_analysis.py --item r3_1
python CODE/r3_analysis.py --item r3_2
python CODE/r3_analysis.py --item a3
python CODE/r3_analysis.py --item r3_3
python CODE/r3_analysis.py --item b4
python CODE/r3_analysis.py --item r3_8
python CODE/r3_analysis.py --item r3_6 --testbed belebele
python CODE/r3_analysis.py --item r3_6 --testbed miracl
python CODE/r3_analysis.py --item r3_7 --part b
python CODE/r3_analysis.py --item r3_7 --part c
python CODE/r3_analysis.py --item c5 --testbed belebele
python CODE/r3_analysis.py --item c5 --testbed haystack
python CODE/r3_analysis.py --item r3_5
python CODE/r3_analysis.py --item r3_4
```

`<tag>`: Belebele, the eight models; haystack, `e5s minilm bgem3 qwen3e_ep`; `r3_miracl.py` with 4 threads,
`minilm bgem3 qwen3e_ep arcticl2`.

### 13. Commercial APIs, vector databases, predicted languages, readers and generated answers (Kaggle)

Each Kaggle script runs as the code of one notebook with internet access (GPU T4 x2 for the reader and judge runs). The
settings at the top of a script select the configuration; the same settings can be passed as environment variables
(`A1_VENDOR`, `A1_TESTBEDS`, `A2_SYSTEM`, `A2_MODEL`, `D6_READER`, `D7_READERS`). Upload each input folder named below
as a Kaggle dataset and attach it. API keys and the Hugging Face token are read from Kaggle Secrets (`VOYAGE_API_KEY`,
`OPENAI_API_KEY`, `HF_TOKEN`); the Llama models require an accepted license on Hugging Face. `a1_api_embed.py`,
`a2_vector_db.py`, `d6_generate.py`, `d6_score.py`, `d7_reader.py` and `scale_taskB_e5s.py` save their work in units
and continue from the output of an earlier run when it is attached.

**Commercial embedding APIs (A1).**
```
python CODE/tools/export_a1_inputs.py
```
Attach `KAGGLE/inputs_a1/` and run `KAGGLE/a1_api_embed.py` with `VENDOR = "voyage"` and `TESTBEDS = ["belebele"]`, and
`KAGGLE/a1_openai_belebele.py`. Place each output folder `a1/<vendor>/` at `DATA/emb/api/<vendor>/`. Then:
```
python CODE/a1_eval.py gate --gate-tag bgem3
python CODE/a1_eval.py voyage --testbeds belebele
python CODE/a1_eval.py --decide
python CODE/a1_eval.py openai --testbeds belebele
python CODE/a1_check_diag.py openai
python CODE/tools/a1_mteb_check.py
```
The first decision covers the Voyage formats. For the decision over both vendors, copy `RESULTS/r3/a1/voyage/` and
`RESULTS/r3/a1/openai/` into `RESULTS/r3/a1/with_openai_20261023/` and run
`python CODE/a1_eval.py --decide --out RESULTS/r3/a1/with_openai_20261023`.

**Vector databases (A2).** Run `KAGGLE/a2_vector_db.py` ten times, with `SYSTEM` one of `chroma`, `elasticsearch`,
`lancedb`, `pgvector`, `qdrant` and `MODEL` either `bgem3` (attach `KAGGLE/inputs_taskB/` from stage 6) or `e5s`
(attach `KAGGLE/inputs_scale_e5s/`). Place each output folder `a2/<system>_<model>/` at
`RESULTS/a2/kaggle/<system>_<model>/`, then run `python CODE/a2_eval.py`.

**Predicted languages (S8).**
```
python CODE/tools/export_s8_inputs.py
```
Attach `KAGGLE/inputs_s8/` and run `KAGGLE/s8_glotlid.py`. Place its output (`lid.parquet`, `info.json`) in
`RESULTS/s8/kaggle/`, then run `python CODE/s8_eval.py` (after `joshi_classes.py`, stage 12).

**Second reader and three passages (D7).**
```
python CODE/r3_d7_prep.py
```
Attach `KAGGLE/inputs_d7/` and run `KAGGLE/d7_reader.py`. Place the downloaded output (the folder that contains `d7/`)
at `RESULTS/r3/d7/kaggle_extract_final_20261024/`, then run
`python CODE/d7_eval.py --src RESULTS/r3/d7/kaggle_extract_final_20261024`.

**Generated answers (D6).**
```
python CODE/tools/d6_query_list.py
python CODE/r3_d6_prep.py bgem3 --part e2
python CODE/r3_d6_prep.py bgem3 --part hnsw
python CODE/r3_d6_prep.py bgem3 --part alloc
MISSION_THREADS=12 python CODE/r3_d6_prep.py e5s --part e2
python CODE/r3_d6_prep.py e5s --part hnsw
MISSION_THREADS=12 python CODE/r3_d6_prep.py e5s --part alloc
python CODE/tools/export_d6_inputs.py
```
`d6_query_list.py` writes the query set to `EXPERIMENTS/d6_queries.csv`. Attach `KAGGLE/inputs_d6/` and run
`KAGGLE/d6_generate.py` once with `READER = "qwen7b"` and once with `READER = "llama8b"`. Place the downloaded outputs
(the folders that contain `d6gen/`) at `RESULTS/r3/d6/gen_extract_qwen7b_20261023/` and
`RESULTS/r3/d6/gen_extract_llama8b_20261024/`.

`python KAGGLE/build_d6_score.py` writes three notebooks next to it: `d6_score.ipynb` (features, validation gate and
judge), `d6_score_smoke.ipynb` (a short test on the gate answers of Bengali and English) and `d6_judge_rerun.ipynb` (the judge again with another
seed). Each runs `d6_score.py` in a separate Python 3.12 environment. Run `d6_score.ipynb` with `KAGGLE/inputs_d6/` and
both generation outputs attached and place its output (the folder that contains `d6score/`) at
`RESULTS/r3/d6/score_v1_20261024/`; run `d6_judge_rerun.ipynb` and place its output (the folder that contains
`d6judge2/`) at `RESULTS/r3/d6/judge_rerun_20261025/`. Then:
```
python CODE/tools/d6_release_consistency.py
python EXPERIMENTS/diagnostics/d6_gate_20261024/d6_gate_diag_summary.py
python EXPERIMENTS/diagnostics/d6_judge_rerun_20261025/d6_judge_selfagreement.py
```
`CODE/d6_eval.py` applies the decision rules of the generated-answer study and stops unless the validation gate
(`d6score/gate.json`) passed. In the reported run the gate did not pass, so the study ends with the gate and the two
diagnostics.

### 14. Numbers, tables and figures of the paper

```
python MANUSCRIPT/analysis/ledger.py
python MANUSCRIPT/paper/figure_scripts/tables.py
python MANUSCRIPT/paper/figure_scripts/tables_r3.py
python MANUSCRIPT/paper/figure_scripts/fig_e1.py
python MANUSCRIPT/paper/figure_scripts/fig_e1_dist.py
python MANUSCRIPT/paper/figure_scripts/fig_family.py
python MANUSCRIPT/paper/figure_scripts/fig_scale.py
python MANUSCRIPT/paper/figure_scripts/fig_a2.py
python MANUSCRIPT/paper/figure_scripts/fig_m7.py
python MANUSCRIPT/paper/figure_scripts/fig_rerank.py
python MANUSCRIPT/paper/figure_scripts/fig_thresholds.py
python MANUSCRIPT/paper/figure_scripts/fig_e6.py
python MANUSCRIPT/paper/figure_scripts/fig_lcfc.py
```

`ledger.py` (with `ledger_r3.py`) computes every number of the paper from `RESULTS/`, the two diagnostics, a few
`DATA/` info files and the haystack constants of `CODE/build_haystack.py`, and writes `MANUSCRIPT/results_ledger.csv` (value, rounding, source file and claim of each number)
and `MANUSCRIPT/paper/generated/results.tex` (one LaTeX macro per number). The table scripts write LaTeX tables to
`MANUSCRIPT/paper/generated/`; the figure scripts write PDF figures to `MANUSCRIPT/paper/figures/` and PNG previews to
`MANUSCRIPT/analysis/_figpng/`.
