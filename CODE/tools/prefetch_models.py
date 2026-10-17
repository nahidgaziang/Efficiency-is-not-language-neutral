import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import ROOT
from huggingface_hub import HfApi, snapshot_download


def main():
    api = HfApi()
    for repo in sys.argv[1:]:
        t0 = time.time()
        info = api.model_info(repo, files_metadata=True)
        path = snapshot_download(repo, revision=info.sha)
        bad = []
        for s in info.siblings:
            f = os.path.join(path, s.rfilename)
            if not os.path.exists(f) or (s.size is not None and os.path.getsize(f) != s.size):
                bad.append((s.rfilename, s.size, os.path.getsize(f) if os.path.exists(f) else None))
        total = sum(s.size or 0 for s in info.siblings)
        print(f"{repo} @ {info.sha[:10]}: {len(info.siblings)} files, {total / 2**20:,.0f} MB, "
              f"{'ALL SIZES MATCH' if not bad else 'MISMATCH ' + str(bad)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
