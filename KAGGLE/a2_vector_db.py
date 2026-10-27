SYSTEM = "qdrant"
MODEL = "bgem3"
MAX_HOURS = 11.0

import os, sys, json, glob, time, hashlib, shutil, subprocess, tarfile, urllib.request
t0 = time.time()
SYSTEM = os.environ.get("A2_SYSTEM", SYSTEM)
MODEL = os.environ.get("A2_MODEL", MODEL)
IN_ROOT = os.environ.get("A2_INPUT_ROOT", "/kaggle/input")
WORK = os.path.join(os.environ.get("A2_OUT", "/kaggle/working/a2"), f"{SYSTEM}_{MODEL}")
TMP = os.environ.get("A2_TMP", "/kaggle/tmp/a2")
FAKE = SYSTEM == "fake"
for d in (WORK, os.path.join(WORK, "units"), TMP):
    os.makedirs(d, exist_ok=True)
def log(msg):
    line = f"[{time.time() - t0:7.0f}s] {msg}"
    print(line, flush=True)
    open(os.path.join(WORK, "run_log.txt"), "a").write(line + "\n")
def hours_left():
    return MAX_HOURS - (time.time() - t0) / 3600
def sh(cmd, check=True, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    log(f"$ {cmd[:160]} -> rc {r.returncode}")
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd[:160]}\n{r.stderr[-1500:]}")
    return r
def pip(*pkgs):
    if os.environ.get("A2_SKIP_PIP"):
        return
    sh(f"{sys.executable} -m pip install -q " + " ".join(pkgs))
def sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()
pip("faiss-cpu==1.15.1")
import numpy as np, pandas as pd, faiss
assert faiss.__version__ == "1.15.1", faiss.__version__
faiss.omp_set_num_threads(os.cpu_count())
log(f"system {SYSTEM}, model {MODEL}; faiss {faiss.__version__}, numpy {np.__version__}, pandas {pd.__version__}, cpus {os.cpu_count()}")

DIM = {"bgem3": 1024, "e5s": 384}[MODEL]
cands = []
for p in glob.glob(os.path.join(IN_ROOT, "**", "e2_P_fp16.npy"), recursive=True):
    d0 = os.path.dirname(p)
    if np.load(p, mmap_mode="r").shape[1] == DIM:
        cands.append(d0)
assert len(cands) >= 1, f"the {MODEL} input dataset is not attached ({'mission-taskb-inputs' if MODEL == 'bgem3' else 'mission-scale-e5s-inputs'})"
INP = cands[0]
mf = [f for f in ("manifest.json", "taskB_e5s_manifest.json") if os.path.exists(os.path.join(INP, f))][0]
man = json.load(open(os.path.join(INP, mf)))
STATE_F = os.path.join(WORK, "state.json")
prev = [p for p in glob.glob(os.path.join(IN_ROOT, "**", "a2", f"{SYSTEM}_{MODEL}", "state.json"), recursive=True)]
if prev and not os.path.exists(STATE_F):
    pd_ = os.path.dirname(prev[0])
    for f in glob.glob(os.path.join(pd_, "units", "*")):
        shutil.copy2(f, os.path.join(WORK, "units", os.path.basename(f)))
    for f in ("state.json", "configs.json"):
        if os.path.exists(os.path.join(pd_, f)):
            shutil.copy2(os.path.join(pd_, f), os.path.join(WORK, f))
    log(f"resumed from {pd_}: {len(os.listdir(os.path.join(WORK, 'units')))} unit files")
state = json.load(open(STATE_F)) if os.path.exists(STATE_F) else {"system": SYSTEM, "model": MODEL, "runs": []}
state["runs"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S")})
def save_state():
    json.dump(state, open(STATE_F + ".tmp", "w"), indent=1); os.replace(STATE_F + ".tmp", STATE_F)
CFG_F = os.path.join(WORK, "configs.json")
configs = json.load(open(CFG_F)) if os.path.exists(CFG_F) else {}
def save_cfg(name, cfg):
    configs[name] = cfg
    json.dump(configs, open(CFG_F + ".tmp", "w"), indent=1, default=str); os.replace(CFG_F + ".tmp", CFG_F)
if not state.get("checksums_ok"):
    for f, v in man.items():
        if not f.startswith("_") and f in ("e2_P_fp16.npy", "e2_Q.npy", "ids.npz", "qrels.json", "e2_per_query_B.parquet"):
            assert sha(os.path.join(INP, f)) == v["sha256"], f"checksum mismatch for {f}"
    state["checksums_ok"] = True; save_state()
ids = np.load(os.path.join(INP, "ids.npz"))
p_id, q_lang, q_id = ids["p_id"], ids["q_lang"], ids["q_id"]
qrels = {tuple(k.split("\t")): set(v) for k, v in json.load(open(os.path.join(INP, "qrels.json"))).items()}
Q = np.load(os.path.join(INP, "e2_Q.npy")).astype(np.float32)
P = np.asarray(np.load(os.path.join(INP, "e2_P_fp16.npy"), mmap_mode="r"), dtype=np.float32)
if os.environ.get("A2_LIMIT"):
    n = int(os.environ["A2_LIMIT"]); P, p_id = P[:n], p_id[:n]
    sel = np.random.default_rng(1).choice(len(Q), 500, replace=False); Q, q_lang, q_id = Q[sel], q_lang[sel], q_id[sel]
qkeys = [(l, q) for l, q in zip(q_lang, q_id)]
log(f"inputs {INP} ({mf}); P {P.shape}, Q {Q.shape}")

disc = 1.0 / np.log2(np.arange(2, 12))
def metrics(I, pids, keys, ref10):
    nd, ov = np.zeros(len(I)), np.zeros(len(I))
    for i, qk in enumerate(keys):
        rel = qrels[qk]
        hits = np.array([(pids[j] in rel) if j >= 0 else False for j in I[i, :10]])
        nd[i] = (hits * disc[:len(hits)]).sum() / disc[:min(len(rel), 10)].sum()
        ov[i] = len(set(I[i, :10].tolist()) & set(ref10[i].tolist())) / 10
    return nd, ov
def unit_done(name):
    return os.path.exists(os.path.join(WORK, "units", f"{name}.parquet"))
def save_unit(name, I, ref10, extra):
    nd, ov = metrics(I, p_id, qkeys, ref10)
    np.save(os.path.join(WORK, "units", f"{name}_ids.npy"), I.astype(np.int32))
    f = os.path.join(WORK, "units", f"{name}.parquet")
    pd.DataFrame({"lang": q_lang, "qid": q_id, "ndcg10": nd, "ov10": ov}).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
    state.setdefault("units", {})[name] = {"mean_ndcg10": float(nd.mean()), "mean_ov10": float(ov.mean()), **extra}
    save_state(); log(f"unit {name}: mean nDCG@10 {nd.mean():.4f}, ov10 {ov.mean():.4f} {extra}")
ex_f = os.path.join(WORK, "units", "exact_ids.npy")
if not unit_done("exact"):
    ix = faiss.IndexFlatIP(DIM); ix.add(P); _, Iex = ix.search(Q, 100); del ix
    Iex = Iex[:, :10]
    save_unit("exact", Iex, Iex, {})
Iex = np.load(ex_f)
if "gate" not in state and not os.environ.get("A2_LIMIT"):
    e2 = pd.read_parquet(os.path.join(INP, "e2_per_query_B.parquet"))
    e2 = e2[e2.method == "exact"].set_index(["lang", "qid"]).reindex(qkeys)
    ours = pd.read_parquet(os.path.join(WORK, "units", "exact.parquet"))
    dmax = float(np.abs(ours.ndcg10.to_numpy() - e2.ndcg10.to_numpy()).max())
    state["gate"] = {"exact_ndcg10_max_abs_diff_vs_E2": dmax, "pass": bool(dmax <= 1e-9)}; save_state()
    log(f"gate: {state['gate']}")
assert os.environ.get("A2_LIMIT") or state["gate"]["pass"], "exact search does not reproduce E2 (max |diff| > 1e-9); stop and send the log"

def topk_fill(rows, n):
    out = -np.ones((len(rows), 10), np.int64)
    for i, r in enumerate(rows):
        r = [int(x) for x in r][:10]
        out[i, :len(r)] = r
    return out

class Fake:
    def setup(self):
        return {"version": "fake"}
    def builds(self):
        return {"hnsw": {}}
    def build(self, name, X):
        h = faiss.IndexHNSWFlat(X.shape[1], 16, faiss.METRIC_INNER_PRODUCT); h.add(X)
        return h, {"type": "faiss HNSWFlat M16 (test)"}
    def settings(self):
        return [("hnsw_ef16", "hnsw", {"ef": 16}), ("hnsw_ef64", "hnsw", {"ef": 64})]
    def search(self, h, Qs, prm):
        h.hnsw.efSearch = prm["ef"]; return h.search(Qs, 10)[1]
    def drop(self, h):
        pass

class Qdrant:
    def setup(self):
        pip("qdrant-client")
        url = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz"
        tgz = os.path.join(TMP, "qdrant.tgz"); urllib.request.urlretrieve(url, tgz)
        tarfile.open(tgz).extractall(TMP)
        env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=os.path.join(TMP, "qdrant_storage"), QDRANT__TELEMETRY_DISABLED="true")
        self.proc = subprocess.Popen([os.path.join(TMP, "qdrant")], env=env, stdout=open(os.path.join(TMP, "qdrant.log"), "w"), stderr=subprocess.STDOUT)
        from qdrant_client import QdrantClient, models
        self.m = models
        for _ in range(120):
            try:
                self.c = QdrantClient(url="http://localhost:6333", timeout=600); v = self.c.info(); break
            except Exception:
                time.sleep(1)
        import qdrant_client
        return {"server": str(v), "client": qdrant_client.__version__ if hasattr(qdrant_client, "__version__") else "unknown"}
    def builds(self):
        m = self.m
        return {"float": None, "sq": m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8)),
                "bq": m.BinaryQuantization(binary=m.BinaryQuantizationConfig()),
                "pq": m.ProductQuantization(product=m.ProductQuantizationConfig(compression=m.CompressionRatio.X16))}
    def build(self, name, X):
        m, coll = self.m, f"a2_{name}_{len(X)}"
        if self.c.collection_exists(coll):
            self.c.delete_collection(coll)
        self.c.create_collection(coll, vectors_config=m.VectorParams(size=X.shape[1], distance=m.Distance.DOT),
                                 quantization_config=self.builds()[name])
        self.c.upload_collection(coll, vectors=X, ids=list(range(len(X))), batch_size=256, parallel=1, wait=True)
        green = 0
        while True:
            ci = self.c.get_collection(coll)
            is_green = str(ci.status).lower().endswith("green")
            if is_green and (ci.indexed_vectors_count or 0) >= len(X) * 0.99:
                break
            green = green + 1 if is_green else 0
            if green >= 6:
                break
            time.sleep(10)
        return coll, json.loads(ci.model_dump_json()) if hasattr(ci, "model_dump_json") else str(ci)
    def settings(self):
        if not hasattr(self, "m"):
            pip("qdrant-client")
            from qdrant_client import models
            self.m = models
        m = self.m
        return [("hnsw_float", "float", None), ("sq_int8", "sq", None),
                ("sq_int8_rescore", "sq", m.QuantizationSearchParams(rescore=True)),
                ("bq_rescore_os1", "bq", m.QuantizationSearchParams(rescore=True, oversampling=1.0)),
                ("bq_rescore_os3", "bq", m.QuantizationSearchParams(rescore=True, oversampling=3.0)),
                ("pq_x16", "pq", None)]
    def search(self, coll, Qs, prm):
        m = self.m
        sp = m.SearchParams(quantization=prm) if prm is not None else None
        out = []
        for a in range(0, len(Qs), 64):
            reqs = [m.QueryRequest(query=q.tolist(), limit=10, params=sp) for q in Qs[a:a + 64]]
            out += [[p.id for p in r.points] for r in self.c.query_batch_points(coll, requests=reqs)]
        return topk_fill(out, 10)
    def drop(self, coll):
        self.c.delete_collection(coll)

class Elastic:
    def setup(self):
        rel = json.load(urllib.request.urlopen("https://api.github.com/repos/elastic/elasticsearch/releases/latest"))
        ver = rel["tag_name"].lstrip("v")
        url = f"https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-{ver}-linux-x86_64.tar.gz"
        tgz = os.path.join(TMP, "es.tgz"); urllib.request.urlretrieve(url, tgz)
        tarfile.open(tgz).extractall(TMP)
        home = os.path.join(TMP, f"elasticsearch-{ver}")
        sh("id esuser || useradd -m esuser", check=False)
        sh(f"chown -R esuser {home}")
        heap = "8g"
        sh(f"su esuser -c 'cd {home} && ES_JAVA_OPTS=\"-Xms{heap} -Xmx{heap}\" ./bin/elasticsearch -d -p pid "
           f"-Expack.security.enabled=false -Ediscovery.type=single-node -Expack.ml.enabled=false'")
        pip(f"elasticsearch=={ver.split('.')[0]}.*")
        from elasticsearch import Elasticsearch, helpers
        self.h = helpers
        for _ in range(180):
            try:
                self.c = Elasticsearch("http://localhost:9200", request_timeout=600); info = self.c.info(); break
            except Exception:
                time.sleep(2)
        lic = self.c.license.get()
        return {"server_version": ver, "info": dict(info), "license": dict(lic)}
    def builds(self):
        return {"hnsw": "hnsw", "int8_hnsw": "int8_hnsw", "int4_hnsw": "int4_hnsw", "bbq_hnsw": "bbq_hnsw"}
    def build(self, name, X):
        idx = f"a2_{name}_{len(X)}"
        self.c.options(ignore_status=404).indices.delete(index=idx)
        self.c.indices.create(index=idx, mappings={"properties": {"v": {"type": "dense_vector", "dims": X.shape[1], "index": True,
                                                                         "index_options": {"type": name}}}})
        def gen():
            for i in range(len(X)):
                yield {"_index": idx, "_id": str(i), "_source": {"v": X[i].tolist()}}
        for ok, info in self.h.streaming_bulk(self.c, gen(), chunk_size=500, request_timeout=600):
            assert ok, info
        self.c.indices.refresh(index=idx)
        mp = self.c.indices.get_field_mapping(index=idx, fields="v", include_defaults=True)
        segs = self.c.cat.segments(index=idx, format="json")
        return idx, {"mapping": dict(mp), "segments": len(segs), "count": self.c.count(index=idx)["count"]}
    def settings(self):
        return [(f"{b}_nc{tag}", b, nc) for b in self.builds() for tag, nc in (("default", None), ("150", 150))]
    def search(self, idx, Qs, nc):
        out = []
        for a in range(0, len(Qs), 100):
            body = []
            for q in Qs[a:a + 100]:
                knn = {"field": "v", "query_vector": q.tolist(), "k": 10}
                if nc is not None:
                    knn["num_candidates"] = nc
                body += [{"index": idx}, {"knn": knn, "size": 10, "_source": False}]
            res = self.c.msearch(searches=body)
            out += [[int(h["_id"]) for h in r["hits"]["hits"]] for r in res["responses"]]
        return topk_fill(out, 10)
    def drop(self, idx):
        self.c.options(ignore_status=404).indices.delete(index=idx)

class PgVector:
    def setup(self):
        sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq postgresql postgresql-contrib git build-essential")
        pgv = sh("ls /usr/lib/postgresql").stdout.split()[-1]
        sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq postgresql-server-dev-{pgv}")
        sh(f"cd {TMP} && rm -rf pgvector && git clone -q --branch v0.8.7 https://github.com/pgvector/pgvector.git && cd pgvector && make -s && make -s install")
        sh("service postgresql start")
        sh("su postgres -c \"psql -c \\\"ALTER USER postgres PASSWORD 'a2';\\\"\"")
        pip("psycopg[binary]", "pgvector")
        import psycopg
        from pgvector.psycopg import register_vector
        self.c = psycopg.connect("host=localhost user=postgres password=a2 dbname=postgres", autocommit=True)
        self.c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.c)
        ver = self.c.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        pg = self.c.execute("SELECT version()").fetchone()[0]
        self.c.execute("SET maintenance_work_mem = '8GB'")
        self.c.execute("SET max_parallel_maintenance_workers = 3")
        return {"pgvector": ver, "postgres": pg, "build_settings": {"maintenance_work_mem": "8GB", "max_parallel_maintenance_workers": 3}}
    def builds(self):
        return {"vector": "vector", "halfvec": "halfvec", "bit": "bit"}
    def build(self, name, X):
        d, t = X.shape[1], f"a2_{len(X)}"
        if not self.c.execute("SELECT to_regclass(%s)", (t,)).fetchone()[0]:
            self.c.execute(f"CREATE TABLE {t} (id int PRIMARY KEY, embedding vector({d}))")
            with self.c.cursor().copy(f"COPY {t} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int4", "vector"])
                for i in range(len(X)):
                    cp.write_row((i, X[i]))
        expr = {"vector": "embedding vector_ip_ops", "halfvec": f"(embedding::halfvec({d})) halfvec_ip_ops",
                "bit": f"(binary_quantize(embedding)::bit({d})) bit_hamming_ops"}[name]
        ix = f"{t}_{name}_hnsw"
        self.c.execute(f"DROP INDEX IF EXISTS {ix}")
        self.c.execute(f"CREATE INDEX {ix} ON {t} USING hnsw ({expr})")
        info = self.c.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s", (ix,)).fetchone()[0]
        return (t, name, d), {"indexdef": info, "hnsw_defaults_note": "m and ef_construction not set (pgvector defaults 16 and 64)"}
    def settings(self):
        return [(f"{b}_ef{ef}", b, ef) for b in ("vector", "halfvec", "bit") for ef in (40, 100)]
    def search(self, h, Qs, ef):
        t, name, d = h
        self.c.execute(f"SET hnsw.ef_search = {ef}")
        if name == "vector":
            sql = f"SELECT id FROM {t} ORDER BY embedding <#> %s LIMIT 10"
        elif name == "halfvec":
            sql = f"SELECT id FROM {t} ORDER BY embedding::halfvec({d}) <#> %s::halfvec({d}) LIMIT 10"
        else:
            sql = (f"SELECT id FROM (SELECT id, embedding FROM {t} ORDER BY binary_quantize(embedding)::bit({d}) <~> "
                   f"binary_quantize(%s::vector) LIMIT 40) c ORDER BY c.embedding <#> %s::vector LIMIT 10")
        if not configs.get(f"explain_{name}"):
            plan = self.c.execute("EXPLAIN " + sql, (Qs[0],) if name != "bit" else (Qs[0], Qs[0])).fetchall()
            save_cfg(f"explain_{name}", [r[0] for r in plan])
        out = []
        for q in Qs:
            out.append([r[0] for r in self.c.execute(sql, (q,) if name != "bit" else (q, q)).fetchall()])
        return topk_fill(out, 10)
    def drop(self, h):
        t, name, _ = h
        self.c.execute(f"DROP INDEX IF EXISTS {t}_{name}_hnsw")

class Chroma:
    def setup(self):
        pip("chromadb")
        import chromadb
        self.c = chromadb.PersistentClient(path=os.path.join(TMP, "chroma"))
        return {"chromadb": chromadb.__version__}
    def builds(self):
        return {"hnsw": None}
    def build(self, name, X):
        coll = f"a2_{len(X)}"
        try:
            self.c.delete_collection(coll)
        except Exception:
            pass
        col = self.c.create_collection(coll, embedding_function=None)
        bs = 5000
        for a in range(0, len(X), bs):
            col.add(ids=[str(i) for i in range(a, min(a + bs, len(X)))], embeddings=X[a:a + bs].tolist())
        cfg = getattr(col, "configuration_json", None) or getattr(col, "metadata", None)
        return col, {"configuration": cfg, "count": col.count()}
    def settings(self):
        return [("hnsw_default", "hnsw", None)]
    def search(self, col, Qs, prm):
        out = []
        for a in range(0, len(Qs), 200):
            r = col.query(query_embeddings=Qs[a:a + 200].tolist(), n_results=10, include=["distances"])
            out += [[int(x) for x in ids] for ids in r["ids"]]
        return topk_fill(out, 10)
    def drop(self, col):
        pass

class Lance:
    def setup(self):
        pip("lancedb", "pyarrow")
        import lancedb, pyarrow as pa
        self.pa = pa
        self.db = lancedb.connect(os.path.join(TMP, "lancedb"))
        return {"lancedb": lancedb.__version__, "pyarrow": pa.__version__}
    def builds(self):
        return {"ivfpq": None}
    def build(self, name, X):
        pa = self.pa
        t = f"a2_{len(X)}"
        tbl = pa.table({"id": pa.array(np.arange(len(X), dtype=np.int64)),
                        "vector": pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1)), X.shape[1])})
        tb = self.db.create_table(t, data=tbl, mode="overwrite")
        tb.create_index(vector_column_name="vector", index_type="IVF_PQ")
        return tb, {"indices": [str(i) for i in tb.list_indices()], "rows": tb.count_rows()}
    def settings(self):
        return [("ivfpq_default", "ivfpq", None), ("ivfpq_refine10", "ivfpq", 10)]
    def search(self, tb, Qs, rf):
        out = []
        for q in Qs:
            s = tb.search(q).limit(10).select(["id"])
            if rf is not None:
                s = s.refine_factor(rf)
            out.append(s.to_pandas()["id"].tolist())
        return topk_fill(out, 10)
    def drop(self, tb):
        pass

S = {"fake": Fake, "qdrant": Qdrant, "elasticsearch": Elastic, "pgvector": PgVector, "chroma": Chroma, "lancedb": Lance}[SYSTEM]()
todo = [s for s in S.settings() if not unit_done(s[0])]
if todo:
    sysinfo = S.setup()
    save_cfg("_system", sysinfo); log(f"system: {json.dumps(sysinfo, default=str)[:600]}")

if todo and "smoke_ok" not in state:
    n = min(20_000, len(P)); Xs = P[:n]; Qs = Q[:500]
    ix = faiss.IndexFlatIP(DIM); ix.add(Xs); _, Ir = ix.search(Qs, 10); del ix
    res = {}
    for b in S.builds():
        h, cfg = S.build(b, Xs)
        for name, bb, prm in S.settings():
            if bb != b:
                continue
            I = S.search(h, Qs, prm)
            ov = float(np.mean([len(set(I[i].tolist()) & set(Ir[i].tolist())) / 10 for i in range(len(Qs))]))
            res[name] = ov
        S.drop(h)
    state["smoke"] = res; save_state(); log(f"smoke (20k passages, 500 queries) ov10: {res}")
    def binary_reference(ncand):
        pb, qb = 2.0 * (Xs > 0) - 1, 2.0 * (Qs > 0) - 1
        Sb = qb @ pb.T + np.random.default_rng(7).uniform(0, 1, (len(Qs), len(Xs)))
        c = np.argpartition(-Sb, ncand, axis=1)[:, :ncand]
        r = np.take_along_axis(Qs @ Xs.T, c, 1)
        top = np.take_along_axis(c, np.argsort(-r, 1)[:, :10], 1)
        return float(np.mean([len(set(top[i].tolist()) & set(Ir[i].tolist())) / 10 for i in range(len(Qs))]))
    binary = {k: 10 * int(k.rsplit("os", 1)[1]) for k in res if k.startswith("bq_rescore_os")}
    binary.update({k: 40 for k in res if k.startswith("bit_ef")})
    ref = {k: binary_reference(n) for k, n in binary.items()}
    state["smoke_binary_reference"] = ref; save_state(); log(f"smoke binary reference (brute force, same subset): {ref}")
    must = [k for k in res if k not in binary and any(s in k for s in ("float", "rescore", "_ef100", "nc150", "hnsw_default", "refine10", "ef64"))]
    assert all(res[k] >= 0.5 for k in must), f"smoke check failed: {res}"
    assert all(res[k] >= 0.5 * ref[k] for k in binary), f"smoke check failed for binary settings: {res} vs reference {ref}"
    state["smoke_ok"] = True; save_state()

for b in S.builds():
    mine = [s for s in S.settings() if s[1] == b and not unit_done(s[0])]
    if not mine:
        continue
    if hours_left() < 1.0:
        log("time budget reached; attach this output and run again"); break
    tb = time.time(); h, cfg = S.build(b, P); bt = time.time() - tb
    save_cfg(b, cfg); log(f"build {b}: {bt:.0f} s")
    for name, _, prm in mine:
        tq = time.time(); I = S.search(h, Q, prm); qt = (time.time() - tq) / len(Q)
        save_cfg(name, {"build": b, "search_param": str(prm)})
        save_unit(name, I, Iex[:, :10], {"build_seconds": round(bt), "seconds_per_query": qt})
    S.drop(h)
complete = all(unit_done(s[0]) for s in S.settings())
if complete:
    json.dump({"system": SYSTEM, "model": MODEL, "settings": [s[0] for s in S.settings()], "limit": os.environ.get("A2_LIMIT"),
               "seconds": round(time.time() - t0)}, open(os.path.join(WORK, "DONE.json"), "w"), indent=1)
log(f"complete = {complete}")
