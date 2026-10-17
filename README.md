# Efficiency is not language-neutral

Code for the study *Efficiency is not language-neutral: auditing compression, approximate search and similarity
thresholds in multilingual dense retrieval*.

Author: Nahid Gazi ([ORCID 0009-0006-1661-9985](https://orcid.org/0009-0006-1661-9985)), Department of Computer Science
and Engineering, Shahjalal University of Science and Technology, Sylhet, Bangladesh

Archived release: [10.5281/zenodo.23434745](https://doi.org/10.5281/zenodo.23434745) (v1.0.0); all versions:
[10.5281/zenodo.23434744](https://doi.org/10.5281/zenodo.23434744).

The code measures how vector compression (product, scalar, binary, rotation-based and residual quantization; PCA and
Matryoshka truncation), approximate nearest-neighbor search (HNSW, IVF) and global similarity thresholds distribute
their retrieval loss across languages. It covers eight embedding models on Belebele (122 language variants), MIRACL (18
languages), Wikipedia haystacks (40 languages and 4 romanized query groups) and a scale test with up to 4.98 million
passages, followed by a downstream RAG reader and a per-language calibration with finite-sample guarantees (LCFC).

## Repository layout

| Path | Content |
|---|---|
| `CODE/` | data preparation, embedding, evaluation and analysis scripts |
| `CODE/tools/` | input export, file validation, independent checks |
| `KAGGLE/` | scripts that ran on Kaggle (2 x NVIDIA T4) |
| `DATA/` | created at run time: datasets, embeddings (not in the repository) |
| `RESULTS/` | created at run time: all outputs (not in the repository) |

Scripts locate `DATA/` and `RESULTS/` relative to the repository root. Most scripts refuse to overwrite an existing
output folder.

## Environment

- Python 3.12.10; packages in `requirements.txt` (PyTorch 2.6.0 with CUDA 12.4 for the embedding steps).
- Embeddings: NVIDIA RTX 3050 (6 GB), fp16, except where stated below. Analyses: CPU (Intel i5-13420H, 15.6 GB RAM).
- Kaggle (2 x T4, 16 GB each): Qwen3-Embedding-4B encoding and the scale test (encoding and the pooled Task B indexes).

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
  the MIRACL evaluation of multilingual-e5-small (`miracl_eval.py e5s`) and the LCFC MIRACL first stages of the same
  model (`lcfc_tables.py fit miracl e5s <stage>`), which used 12 threads.
- HNSW graphs built with several threads are not bit-identical between builds; HNSW results are reproducible in
  distribution, not bit for bit.
- Random seeds are fixed in the scripts (bootstrap seeds 20261006, 20261007 and 20261015; fit seeds 0, 1, 2).

## Data

| Data | Source | Prepared by |
|---|---|---|
| Belebele, test split, 122 language variants | `facebook/belebele`, revision `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` | `CODE/embed_belebele.py` (loaded with `datasets`) |
| MIRACL dev, 18 languages | `mteb/MIRACLRetrievalHardNegatives`, revision `332a9acb49f5e83d5397683f79d23e588f685916` | `CODE/get_miracl.py`; 40,000 passages per language drawn by `CODE/embed_miracl.py` (seed 20261006) |
| Wikipedia haystacks | `wikimedia/wikipedia`, 20231101 dumps | `CODE/build_haystack.py` (20,000 paragraphs per language plus the Belebele gold passages) |
| Scale test | `miracl/miracl-corpus` v1.0, up to 300,000 passages per language | `CODE/tools/export_scale_inputs.py`, then `KAGGLE/scale_miracl_bgem3.py` |

Each dataset and model keeps its own license; see its card on Hugging Face.

## Models

| Tag | Model | Revision |
|---|---|---|
| `e5s` | intfloat/multilingual-e5-small | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `minilm` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| `bgem3` | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` |
| `qwen3e_ep` | Qwen/Qwen3-Embedding-0.6B, developer evaluation prompts | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `qwen3e4b_ep` | Qwen/Qwen3-Embedding-4B, developer evaluation prompts (Kaggle) | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| `arcticl2` | Snowflake/snowflake-arctic-embed-l-v2.0 | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` |
| `egemma` | google/embeddinggemma-300m (bfloat16) | `57c266a740f537b4dc058e1b0cda161fd15afa75` |
| `jinav5` | jinaai/jina-embeddings-v5-text-small-retrieval (CC BY-NC 4.0) | `6856e76bb72982e58de0620458a4e8b3614da340` |
| `qwen3e`, `qwen3e4b` | the two Qwen3 models with an earlier query prompt; used only for the prompt comparison | as above |
| `l3bn` | l3cube-pune/bengali-sentence-similarity-sbert (Bangla case study) | `3af011cee9743eee4034904a97a47a769258d5b8` |
| RAG reader | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

Query and passage prompts are defined in `CODE/common.py` (`MODELS`, `QUERY_PROMPTS`), `CODE/embed_belebele.py` and
`CODE/embed_belebele_new.py`. `python CODE/tools/prefetch_models.py <repo id> ...` downloads a model into the project
cache and checks every file size.

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

Configuration audit of the other models against published MTEB results: `python CODE/tools/config_audit.py`.

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

```
python CODE/tools/export_scale_inputs.py bgem3 <repository root>/KAGGLE/scale_inputs
```
Upload `KAGGLE/scale_inputs/` to Kaggle, run `KAGGLE/scale_miracl_bgem3.py`, and place the output folder `scale_bgem3/`
under `DATA/emb/bgem3/scale/`. Then:
```
python CODE/tools/check_scale_output.py DATA/emb/bgem3/scale
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
