import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THREADS = int(os.environ.get("MISSION_THREADS", "10"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))


def _no_power_throttling():
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        s = _PPTS(1, 0x1, 0x0)
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))
    except Exception:
        pass


_no_power_throttling()
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "DATA", "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
DATA = os.path.join(ROOT, "DATA")
EMB = os.path.join(DATA, "emb")
RESULTS = os.path.join(ROOT, "RESULTS")

MODELS = {
    "e5s": ("intfloat/multilingual-e5-small", "query: ", "passage: ", {}, 512, 128),
    "e5b": ("intfloat/multilingual-e5-base", "query: ", "passage: ", {}, 512, 64),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", {}, 128, 128),
    "labse": ("sentence-transformers/LaBSE", "", "", {}, 256, 64),
    "bgem3": ("BAAI/bge-m3", "", "", {}, 512, 64),
    "gte": ("Alibaba-NLP/gte-multilingual-base", "", "", {"trust_remote_code": True}, 512, 64),
    "qwen3e": ("Qwen/Qwen3-Embedding-0.6B",
               "Instruct: Given a question, retrieve passages that answer the question\nQuery: ", "", {}, 512, 16),
    "l3bn": ("l3cube-pune/bengali-sentence-similarity-sbert", "", "", {}, 512, 64),
    "qwen3e4b": ("Qwen/Qwen3-Embedding-4B",
                 "Instruct: Given a question, retrieve passages that answer the question\nQuery: ", "", {}, 512, 16),
    "qwen3e_ep": ("Qwen/Qwen3-Embedding-0.6B",
                  "Instruct: Retrieval the relevant passage for the given query\nQuery:", "", {}, 512, 16),
    "qwen3e4b_ep": ("Qwen/Qwen3-Embedding-4B",
                    "Instruct: Retrieval the relevant passage for the given query\nQuery:", "", {}, 512, 16),
    "arcticl2": ("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", "",
                 {"revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a", "local_files_only": True}, 512, 64),
}

QUERY_PROMPTS = {"qwen3e_ep": {"miracl40k": "Instruct: Retrieval relevant passage for the given query\nQuery:",
                               "belebele": "Instruct: Retrieval the relevant passage for the given query\nQuery:",
                               "haystack": "Instruct: Retrieval the relevant passage for the given query\nQuery:"}}

MRL_MODELS = {"gte": (128, 256, 512), "qwen3e": (128, 256, 512), "qwen3e4b": (128, 256, 512, 1024),
              "qwen3e_ep": (128, 256, 512), "qwen3e4b_ep": (128, 256, 512, 1024),
              "arcticl2": (256,), "egemma": (128, 256, 512), "jinav5": (32, 64, 128, 256, 512, 768)}

CROSS_SCRIPT = {"ben_Latn": "ben_Beng", "hin_Latn": "hin_Deva", "urd_Latn": "urd_Arab", "npi_Latn": "npi_Deva"}

MIRACL_LANGS = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh"]


def tie_aware_recall(S, gold, k):
    import numpy as np
    sg = S[np.arange(len(gold)), gold][:, None]
    g = (S > sg).sum(1)
    t = (S == sg).sum(1) - 1
    return np.clip((k - g) / (t + 1), 0, 1)
