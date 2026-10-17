import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import RESULTS, MIRACL_LANGS

d = json.load(open(os.path.join(RESULTS, "integrity", "qwen3_prompt_check_qwen3e_miracl.json")))
rows = {r["lang"]: r for r in d["per_lang"]}
assert sorted(rows) == sorted(MIRACL_LANGS), "not all 18 languages present"
gain = {L: r["V4_qwen_eval_miracl"]["ndcg10"] - r["stored"]["ndcg10"] for L, r in rows.items()}
n_up = sum(g > 0 for g in gain.values())
mean_gain = sum(gain.values()) / len(gain)
ok = n_up >= 15 and mean_gain >= 0.10
res = {"n_languages": len(gain), "n_V4_above_stored": n_up, "mean_gain": mean_gain, "criterion_met": ok,
       "per_lang": {L: {"stored": rows[L]["stored"]["ndcg10"], "V2_card_default": rows[L]["V2_card_default"]["ndcg10"],
                        "V4_qwen_eval_miracl": rows[L]["V4_qwen_eval_miracl"]["ndcg10"], "gain_V4": gain[L]} for L in MIRACL_LANGS}}
print(json.dumps(res, indent=1))
out = os.path.join(RESULTS, "integrity", "qwen3_prompt_confirm.json")
assert not os.path.exists(out), f"{out} exists"
json.dump(res, open(out, "w"), indent=1)
if ok:
    open(os.path.join(RESULTS, "integrity", "qwen3_prompt_fix_confirmed.PASS"), "w").write(
        f"criterion met: V4 > stored in {n_up}/18 languages, mean gain {mean_gain:.3f}\n")
    print("marker written")
else:
    print("criterion NOT met: no marker; report to the user")
