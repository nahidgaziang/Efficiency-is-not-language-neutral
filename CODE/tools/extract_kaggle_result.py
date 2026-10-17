import os, sys, json, zipfile, hashlib, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import EMB

zp, tag = sys.argv[1], sys.argv[2]
dst = os.path.join(EMB, tag); os.makedirs(dst, exist_ok=True)
z = zipfile.ZipFile(zp)
for name in (f"{tag}/belebele.npz", f"{tag}/belebele_info.json"):
    out = os.path.join(dst, os.path.basename(name))
    if os.path.exists(out):
        sys.exit(f"{out} exists - not overwritten")
    with z.open(name) as src, open(out + ".tmp", "wb") as f:
        shutil.copyfileobj(src, f, length=16 << 20)
    os.replace(out + ".tmp", out)
    print("extracted", out, os.path.getsize(out), "bytes")
info = json.load(open(os.path.join(dst, "belebele_info.json"), encoding="utf-8"))
h = hashlib.sha256()
with open(os.path.join(dst, "belebele.npz"), "rb") as f:
    for b in iter(lambda: f.read(16 << 20), b""):
        h.update(b)
print("sha256", h.hexdigest(), "| matches the notebook's record:", h.hexdigest() == info["npz_sha256"])
sys.exit(0 if h.hexdigest() == info["npz_sha256"] else 1)
