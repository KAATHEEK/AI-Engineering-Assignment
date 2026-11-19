#!/usr/bin/env python3
"""
Task 3 - Run a small model locally in two quantisation settings and compare them.

Model : Qwen2.5-3B-Instruct served by Ollama
Setups: fp16 (reference) and q4_K_M (4-bit)

    python main.py                 # benchmark both settings, judge quality, print calculations
    python main.py --calc-only     # only the KV-cache / 20-user serving calculations (no Ollama, no GPU)
    python main.py --pull          # download the two model tags first (about 8 GB)
    python main.py --skip-judge    # skip the LLM-judge step (no API key needed)

Prerequisite: Ollama installed and running (`ollama serve`). See README.md.
The LLM judge can be any OpenAI-compatible API (Groq, NVIDIA, ...); it only grades
answers, the benchmarked model always runs locally.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA
SETTINGS = {"fp16": "qwen2.5:3b-instruct-fp16", "q4_K_M": "qwen2.5:3b-instruct-q4_K_M"}

PROMPTS = [
    "Explain in 3 sentences why the sky is blue.",
    "A train leaves at 2:15 PM and travels 180 km at 60 km/h. At what time does it arrive? Show your calculation.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Summarize in exactly two sentences: 'The city council voted 7-2 on Tuesday to approve a new network of protected bike lanes covering 40 km of roads. Supporters said it will cut traffic injuries, while opponents argued it will remove parking and hurt local shops. Construction begins in March and is expected to take 18 months.'",
    "Translate to French: 'Could you tell me where the nearest train station is?'",
    "List five differences between a process and a thread as a markdown table.",
    "Return ONLY valid JSON with keys name, age, city for this sentence: Anita is 29 and lives in Pune.",
    "What is 17 x 24? Give just the number.",
    "Write a four-line poem about autumn rain.",
    "Give three pros and three cons of microservices, in under 120 words.",
]
assert len(PROMPTS) == 10


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def ollama_get(path):
    import requests
    return requests.get(f"{OLLAMA}{path}", timeout=30).json()


def ensure_models(pull):
    import requests
    try:
        have = {m["name"]: m["size"] / 1e9 for m in ollama_get("/api/tags")["models"]}
    except Exception:
        sys.exit(f"Cannot reach Ollama at {OLLAMA}. Install it from https://ollama.com and run `ollama serve`.")
    for tag in SETTINGS.values():
        if tag in have:
            continue
        if not pull:
            sys.exit(f"Model {tag} is not downloaded. Run `ollama pull {tag}` or re-run with --pull.")
        print(f"Pulling {tag} (this can take a while) ...")
        requests.post(f"{OLLAMA}/api/pull", json={"model": tag, "stream": False}, timeout=3600).raise_for_status()
    return {m["name"]: m["size"] / 1e9 for m in ollama_get("/api/tags")["models"]}


def gpu_mem_mb():
    """Used GPU memory from nvidia-smi (MB); None when there is no NVIDIA GPU / nvidia-smi."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def loaded_models():
    return ollama_get("/api/ps").get("models", [])


def unload_all():
    import requests
    for m in loaded_models():
        requests.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    time.sleep(5)


def generate(model, prompt, num_predict=256):
    """Same settings for both setups: temperature 0, seed 42, 4096 context, 256 new tokens max."""
    import requests
    r = requests.post(f"{OLLAMA}/api/generate", timeout=900, json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 42, "num_predict": num_predict, "num_ctx": 4096}})
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Step 1 - speed and memory
# --------------------------------------------------------------------------- #
def benchmark(disk_gb):
    import numpy as np
    rows, outputs = [], {}
    for name, tag in SETTINGS.items():
        unload_all()
        mem_before = gpu_mem_mb()
        t0 = time.time()
        generate(tag, "Hello", 8)                       # loads the model onto the GPU
        load_s = time.time() - t0
        mem_after = gpu_mem_mb()
        lm = loaded_models()
        vram = lm[0].get("size_vram", 0) / 1e9 if lm else float("nan")

        tps, pps, outs = [], [], []
        for p in PROMPTS:
            r = generate(tag, p)
            outs.append(r["response"].strip())
            tps.append(r["eval_count"] / (r["eval_duration"] / 1e9))          # decode speed only
            if r.get("prompt_eval_duration"):
                pps.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
        outputs[name] = outs
        rows.append({
            "setting": name, "disk_GB": round(disk_gb.get(tag, float("nan")), 2),
            "GPU_mem_loaded_MB": (mem_after - mem_before) if None not in (mem_before, mem_after) else None,
            "ollama_size_vram_GB": round(vram, 2),
            "decode_tok/s_mean": round(float(np.mean(tps)), 1), "decode_tok/s_std": round(float(np.std(tps)), 1),
            "prompt_tok/s_mean": round(float(np.mean(pps)), 1) if pps else None,
            "load_time_s": round(load_s, 1)})
        print(f"{name} done: {rows[-1]['decode_tok/s_mean']} tok/s")
    return rows, outputs


# --------------------------------------------------------------------------- #
# Step 2 - quality: automatic checks + blind LLM judge
# --------------------------------------------------------------------------- #
def check_train(o): return "5:15" in o or "17:15" in o


def check_json(o):
    s, e = o.find("{"), o.rfind("}")
    try:
        d = json.loads(o[s:e + 1])
        return set(d) >= {"name", "age", "city"} and int(d["age"]) == 29 and "pune" in str(d["city"]).lower()
    except Exception:
        return False


def check_mult(o): return "408" in o


AUTO = {1: check_train, 6: check_json, 7: check_mult}      # 0-based prompt index -> verifier

JUDGE_SYS = ("You are a strict grader. You see a PROMPT and two answers, A and B. Score each answer from 1 to 5 for "
             "correctness, instruction-following and clarity. "
             'Respond ONLY with JSON: {"A": <int>, "B": <int>, "note": "<one sentence>"}')


def get_judge():
    from openai import OpenAI
    key = os.environ.get("LLM_API_KEY")
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if not key:
        if os.environ.get("GROQ_API_KEY"):
            key, base, model = os.environ["GROQ_API_KEY"], base or "https://api.groq.com/openai/v1", model or "openai/gpt-oss-120b"
        elif os.environ.get("NVIDIA_API_KEY"):
            key, base, model = os.environ["NVIDIA_API_KEY"], base or "https://integrate.api.nvidia.com/v1", model or "meta/llama-3.3-70b-instruct"
    if not (key and base and model):
        return None
    return OpenAI(api_key=key, base_url=base), model


def judge(outputs):
    j = get_judge()
    if j is None:
        print("No judge API key set (GROQ_API_KEY / NVIDIA_API_KEY / LLM_API_KEY). Skipping the judge.")
        return []
    client, model = j
    print("Judge model:", model)
    rng, rows = random.Random(0), []
    for i, p in enumerate(PROMPTS):
        flip = rng.random() < 0.5                       # random A/B order so position cannot bias the result
        a, b = ("q4_K_M", "fp16") if flip else ("fp16", "q4_K_M")
        user = f"PROMPT:\n{p}\n\nANSWER A:\n{outputs[a][i]}\n\nANSWER B:\n{outputs[b][i]}"
        scores = None
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}])
                txt = resp.choices[0].message.content
                scores = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
                break
            except Exception as e:
                print("  judge retry:", str(e)[:100])
                time.sleep(10 * (attempt + 1))
        if scores:
            rows.append({"prompt": i + 1, a: scores["A"], b: scores["B"], "note": scores.get("note", "")})
        time.sleep(1)
    return rows


# --------------------------------------------------------------------------- #
# Step 3 - KV-cache and serving calculations
# --------------------------------------------------------------------------- #
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_val=2):
    """2 (K and V) x layers x KV heads x head dim x bytes per value."""
    return 2 * layers * kv_heads * head_dim * bytes_per_val


def weights_gb(params_b, bits, overhead=1.0):
    return params_b * 1e9 * bits / 8 * overhead / 1e9


def calculations(speed_rows=None):
    L = []
    emit = lambda s="": (L.append(s), print(s))

    emit("== KV cache per token (fp16) ==")
    for name, (layers, heads, dim) in {"Qwen2.5-3B  (36 layers, 2 KV heads, d=128)": (36, 2, 128),
                                       "Llama-3.1-8B (32 layers, 8 KV heads, d=128)": (32, 8, 128)}.items():
        b = kv_bytes_per_token(layers, heads, dim)
        emit(f"{name}: {b / 1024:.0f} KB/token -> {b * 4096 / 1e9:.2f} GB for one 4096-token sequence")
    emit("\n== Weights memory by precision ==")
    for bits, ov, label in [(16, 1.0, "fp16"), (8, 1.0, "int8"), (4, 1.1, "4-bit (+~10% scales)")]:
        emit(f"3B model {label:>22}: {weights_gb(3.09, bits, ov):.2f} GB | 8B model: {weights_gb(8.03, bits, ov):.2f} GB")

    # ---- plan: 20 concurrent users on ONE 24 GB GPU ----
    GPU_GB, USABLE = 24, 0.90
    W4, W16, OVERHEAD = 5.7, 16.06, 2.0                 # 4-bit AWQ keeps embeddings in 16-bit -> ~5.7 GB
    USERS, CTX = 20, 4096
    LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128             # Llama-3.1-8B
    BANDWIDTH_GBPS, EFFICIENCY = 1000, 0.6
    kv_tok = kv_bytes_per_token(LAYERS, KV_HEADS, HEAD_DIM)
    kv_total = USERS * CTX * kv_tok / 1e9
    budget = GPU_GB * USABLE
    emit(f"\n== Serving plan: {USERS} users x {CTX} tokens, Llama-3.1-8B class, {GPU_GB} GB GPU ==")
    emit(f"KV per token {kv_tok / 1024:.0f} KB | KV for all users: {kv_total:.1f} GB (fp16) / {kv_total / 2:.1f} GB (fp8)")
    for label, w, kv in [("4-bit AWQ + fp16 KV", W4, kv_total), ("4-bit AWQ + fp8 KV", W4, kv_total / 2),
                         ("fp16 weights + fp16 KV", W16, kv_total), ("fp16 weights + fp8 KV", W16, kv_total / 2)]:
        need = w + kv + OVERHEAD
        emit(f"{label:<24} need {need:5.1f} GB vs budget {budget:.1f} GB -> {'FITS' if need <= budget else 'DOES NOT FIT'}")
    emit(f"Max concurrent {CTX}-token users (4-bit weights, fp16 KV): ~{(budget - W4 - OVERHEAD) * 1e9 / (CTX * kv_tok):.0f}")
    step_s = (W4 + kv_total) * 1e9 / (BANDWIDTH_GBPS * 1e9 * EFFICIENCY)     # memory-bandwidth-bound decode
    emit(f"Decode step ~{step_s * 1000:.0f} ms -> ~{1 / step_s:.0f} tok/s per user, ~{USERS / step_s:.0f} tok/s total")
    emit("(Assumes every user is at the full 4096 tokens, so this is a conservative estimate.)")

    if speed_rows:
        emit("\n== Measured quantisation effect ==")
        s = {r["setting"]: r for r in speed_rows}
        f, q = s["fp16"], s["q4_K_M"]
        emit(f"fp16  : file {f['disk_GB']} GB ({f['disk_GB'] * 8 / 3.09:.1f} bits/weight) | decode {f['decode_tok/s_mean']} tok/s")
        emit(f"q4_K_M: file {q['disk_GB']} GB ({q['disk_GB'] * 8 / 3.09:.1f} bits/weight) | decode {q['decode_tok/s_mean']} tok/s")
        emit(f"File size ratio {f['disk_GB'] / q['disk_GB']:.1f}x | speed ratio {q['decode_tok/s_mean'] / f['decode_tok/s_mean']:.1f}x")
    return L


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calc-only", action="store_true")
    ap.add_argument("--pull", action="store_true", help="download the two Ollama model tags if missing")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.calc_only:
        lines = calculations()
    else:
        import pandas as pd
        disk = ensure_models(args.pull)
        rows, outputs = benchmark(disk)
        speed = pd.DataFrame(rows)
        print("\n", speed.to_string(index=False))
        speed.to_csv(os.path.join(args.out_dir, "benchmark_speed_memory.csv"), index=False)
        json.dump(outputs, open(os.path.join(args.out_dir, "outputs.json"), "w"), indent=1)

        print("\n== Automatic checks (verifiable prompts) ==")
        for idx, fn in AUTO.items():
            print(f"Prompt {idx + 1}:", {s: fn(outputs[s][idx]) for s in SETTINGS})
        same = sum(outputs["fp16"][i].strip() == outputs["q4_K_M"][i].strip() for i in range(len(PROMPTS)))
        print(f"Identical answers: {same}/{len(PROMPTS)} (greedy decoding, so any difference comes from the weights)")

        if not args.skip_judge:
            jrows = judge(outputs)
            if jrows:
                jdf = pd.DataFrame(jrows)
                print(jdf.to_string(index=False))
                print("Mean judge score:", jdf[["fp16", "q4_K_M"]].mean().round(2).to_dict())
                jdf.to_csv(os.path.join(args.out_dir, "judge_scores.csv"), index=False)
        print()
        lines = calculations(rows)

    open(os.path.join(args.out_dir, "calculations.txt"), "w").write("\n".join(lines) + "\n")
    print(f"\nSaved results to {args.out_dir}/")


if __name__ == "__main__":
    main()
