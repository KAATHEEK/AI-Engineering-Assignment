#!/usr/bin/env python3
"""
Task 1 - LoRA fine-tuning of a small open model (Qwen2.5-0.5B-Instruct).

Goal: turn a plain-English meeting request into a fixed JSON schema.

    python main.py                     # all stages, in order
    python main.py --stage data        # build 600 labelled examples
    python main.py --stage train       # LoRA fine-tune, adapter saved to results/
    python main.py --stage evaluate    # base vs fine-tuned on 100 held-out examples
    python main.py --stage memory      # 7B memory estimate: full FT vs LoRA vs QLoRA

Labels are generated programmatically (so they are exact by construction) and an
LLM is only used to phrase each label as a natural request. Any OpenAI-compatible
API works (Groq, NVIDIA build.nvidia.com, OpenAI, ...). See README.md.
"""
import argparse
import datetime
import json
import os
import random
import sys
import time
from collections import Counter

SEED = 42
N_TOTAL, N_TRAIN, N_VAL = 600, 450, 50          # remaining 100 are the held-out test set
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data/all.json"
OUT_DIR = "results"
ADAPTER_DIR = os.path.join(OUT_DIR, "lora-meeting-json")

# Short prompt used for fine-tuning: the model has to LEARN the schema.
SYSTEM_SHORT = "Convert the user's meeting request into JSON."
# Long prompt that spells the schema out: gives the BASE model its best chance.
SYSTEM_SCHEMA = (
    "You convert meeting requests into JSON. Output ONLY a JSON object with exactly these keys: "
    'action ("schedule"|"cancel"|"reschedule"), title (lowercase string), date (YYYY-MM-DD, year 2026), '
    'time (24-hour HH:MM), duration_min (integer minutes), attendees (list of first names), '
    'priority ("low"|"medium"|"high"). No explanations, no markdown.'
)
FIELDS = ["action", "title", "date", "time", "duration_min", "attendees", "priority"]


# --------------------------------------------------------------------------- #
# LLM client (any OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct"),
}


def get_llm():
    """Return (client, model_name). Reads LLM_API_KEY / GROQ_API_KEY / NVIDIA_API_KEY."""
    from openai import OpenAI

    key = os.environ.get("LLM_API_KEY")
    base, model = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_MODEL")
    if not key:
        for env, provider in (("GROQ_API_KEY", "groq"), ("NVIDIA_API_KEY", "nvidia")):
            if os.environ.get(env):
                key = os.environ[env]
                base = base or PROVIDERS[provider][0]
                model = model or PROVIDERS[provider][1]
                break
    if not key:
        sys.exit("No API key found. Set GROQ_API_KEY, NVIDIA_API_KEY, or LLM_API_KEY "
                 "(with LLM_BASE_URL and LLM_MODEL), or run with --no-llm.")
    if not base or not model:
        sys.exit("LLM_API_KEY also needs LLM_BASE_URL and LLM_MODEL.")
    return OpenAI(api_key=key, base_url=base), model


def parse_json_object(text):
    """Pull the first {...} block out of a model reply; None if it is not valid JSON."""
    if not isinstance(text, str):
        return None
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e < s:
        return None
    try:
        obj = json.loads(text[s:e + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Stage 1 - dataset
# --------------------------------------------------------------------------- #
ACTIONS = ["schedule", "cancel", "reschedule"]
TITLES = ["budget review", "design sync", "project kickoff", "sprint planning", "client demo",
          "quarterly planning", "onboarding session", "code review", "team retro", "roadmap discussion",
          "vendor call", "hiring debrief", "security audit", "product launch prep", "performance review"]
NAMES = ["Priya", "Rahul", "Anita", "Karthik", "Sneha", "Vikram", "Meera", "Arjun",
         "Divya", "Imran", "Neha", "Suresh", "Kavya", "Rohan", "Fatima", "Sanjay"]
DURATIONS = [15, 30, 45, 60, 90, 120]
PRIORITIES = ["low", "medium", "high"]

PARA_SYS = (
    "You write realistic messages that a busy person might type to an assistant. "
    "For each JSON spec you receive, write ONE natural request that conveys exactly the same facts. Rules: "
    "mention the action (schedule / cancel / reschedule) naturally; the meeting title; "
    "the date with the month written as a WORD, the day number as digits, and the year 2026; "
    "the time in any natural format (3pm, 15:30, 3:30 in the afternoon); "
    "the duration in any natural phrasing (an hour, 90 minutes, half an hour); "
    "every attendee by first name; and the priority through natural wording "
    "(high = urgent / ASAP / top priority; low = no rush / whenever / low priority; medium = normal / standard / regular). "
    "Vary the style: some terse, some polite, some chatty. Never copy the JSON keys into the request. "
    'Return ONLY a JSON object: {"requests": ["...", "..."]} with exactly one string per spec, in the same order.'
)


def make_spec():
    return {
        "action": random.choice(ACTIONS),
        "title": random.choice(TITLES),
        "date": f"2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "time": f"{random.randint(8, 18):02d}:{random.choice(['00', '15', '30', '45'])}",
        "duration_min": random.choice(DURATIONS),
        "attendees": random.sample(NAMES, random.choice([1, 1, 2, 2, 3])),
        "priority": random.choice(PRIORITIES),
    }


def make_specs(n):
    specs, seen = [], set()
    while len(specs) < n:
        s = make_spec()
        k = json.dumps(s)
        if k not in seen:
            seen.add(k)
            specs.append(s)
    return specs


def fallback_text(s):
    """Template sentence used when the LLM paraphrase is missing or drops a fact."""
    d = datetime.date.fromisoformat(s["date"])
    h, m = map(int, s["time"].split(":"))
    t = f"{(h % 12) or 12}:{m:02d}{'am' if h < 12 else 'pm'}"
    prio = {"high": "This is urgent.", "medium": "Normal priority.", "low": "No rush."}[s["priority"]]
    verb = {"schedule": "Please schedule a", "cancel": "Please cancel the",
            "reschedule": "Please reschedule the"}[s["action"]]
    who = " and ".join(s["attendees"])
    return (f"{verb} {s['title']} with {who} on {d.strftime('%B')} {d.day}, {d.year} "
            f"at {t} for {s['duration_min']} minutes. {prio}")


def text_ok(text, s):
    """Keep an LLM request only if it still contains every attendee, the month and the day."""
    if not isinstance(text, str):
        return False
    t = text.lower()
    d = datetime.date.fromisoformat(s["date"])
    return (all(n.lower() in t for n in s["attendees"])
            and d.strftime("%B").lower() in t and str(d.day) in t)


def paraphrase_batch(client, model, batch, tries=5):
    for attempt in range(tries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.9,
                messages=[{"role": "system", "content": PARA_SYS},
                          {"role": "user", "content": json.dumps(batch)}])
            obj = parse_json_object(resp.choices[0].message.content)
            texts = obj.get("requests") if obj else None
            if isinstance(texts, list) and len(texts) == len(batch):
                return texts
        except Exception as e:                       # rate limit, network, bad reply
            wait = 15 * (attempt + 1)
            print(f"  retry {attempt + 1}/{tries}: {str(e)[:120]} | waiting {wait}s")
            time.sleep(wait)
    return [None] * len(batch)


def stage_data(args):
    if os.path.exists(args.data_path):
        rows = json.load(open(args.data_path))
        print(f"Loaded cached dataset: {len(rows)} rows ({args.data_path})")
        return rows

    random.seed(SEED)
    specs = make_specs(N_TOTAL)
    client = model = None
    if not args.no_llm:
        client, model = get_llm()
        print(f"Paraphrasing with model: {model}")

    rows, n_fallback, batch_size = [], 0, 10
    for i in range(0, len(specs), batch_size):
        batch = specs[i:i + batch_size]
        texts = paraphrase_batch(client, model, batch) if client else [None] * len(batch)
        for s, t in zip(batch, texts):
            if not text_ok(t, s):
                t, n_fallback = fallback_text(s), n_fallback + 1
            rows.append({"request": t.strip(), "target": s})
        if client:
            print(f"  {len(rows)}/{len(specs)} done | template fallbacks so far: {n_fallback}")
            time.sleep(2)

    os.makedirs(os.path.dirname(args.data_path) or ".", exist_ok=True)
    json.dump(rows, open(args.data_path, "w"), indent=1)
    print(f"Saved {len(rows)} rows to {args.data_path} | template fallbacks: {n_fallback}")
    return rows


def split(rows):
    return rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN + N_VAL], rows[N_TRAIN + N_VAL:]


# --------------------------------------------------------------------------- #
# Stage 2 - train
# --------------------------------------------------------------------------- #
def load_tokenizer_and_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. Run on a GPU machine (a free Colab T4 is enough).")
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    except TypeError:                                 # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return tok, model.to("cuda")


def build_prompt(tok, system, request):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": request}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def stage_train(args, rows):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    train, val, _ = split(rows)
    tok, model = load_tokenizer_and_model(args.base_model)
    max_len = 320

    def encode(ex):
        prompt = build_prompt(tok, SYSTEM_SHORT, ex["request"])
        answer = json.dumps(ex["target"]) + "<|im_end|>"
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(answer, add_special_tokens=False)["input_ids"]
        return {"input_ids": (p + a)[:max_len],
                "attention_mask": [1] * len((p + a)[:max_len]),
                "labels": ([-100] * len(p) + a)[:max_len]}        # loss on the JSON answer only

    train_ds = Dataset.from_list([encode(x) for x in train])
    val_ds = Dataset.from_list([encode(x) for x in val])
    print("max sequence length in train:", max(len(x) for x in train_ds["input_ids"]))

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    common = dict(
        output_dir=os.path.join(OUT_DIR, "trainer-tmp"),
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=3, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_steps=10, weight_decay=0.0,
        fp16=True, logging_steps=10, save_strategy="no", report_to="none",
        remove_unused_columns=False, seed=SEED)
    try:
        targs = TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:                                 # transformers < 4.41
        targs = TrainingArguments(evaluation_strategy="epoch", **common)

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    print(f"\nPeak GPU memory during LoRA training: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print("Adapter saved to", ADAPTER_DIR)


# --------------------------------------------------------------------------- #
# Stage 3 - evaluate (base vs fine-tuned) + failure analysis
# --------------------------------------------------------------------------- #
def extract_json(text):
    return parse_json_object(text)


def strict_json(text):
    try:
        return isinstance(json.loads(text), dict)
    except Exception:
        return False


def norm_val(field, v):
    try:
        if field == "attendees":
            return sorted(str(x).strip().lower() for x in v)
        if field == "duration_min":
            return int(v)
        return str(v).strip().lower()
    except Exception:
        return None


def score(preds, golds):
    n = len(golds)
    parsed = [extract_json(p) for p in preds]
    valid = sum(p is not None for p in parsed)
    strict = sum(strict_json(p) for p in preds)
    exact, per = 0, {f: 0 for f in FIELDS}
    for p, g in zip(parsed, golds):
        if p is None:
            continue
        all_ok = True
        for f in FIELDS:
            good = norm_val(f, p.get(f)) == norm_val(f, g[f])
            per[f] += int(good)
            all_ok = all_ok and good
        exact += int(all_ok)
    res = {"strict_json_%": 100 * strict / n, "valid_json_%": 100 * valid / n,
           "exact_match_%": 100 * exact / n,
           "field_acc_%": 100 * sum(per.values()) / (n * len(FIELDS))}
    res.update({f"acc_{f}_%": 100 * per[f] / n for f in FIELDS})
    return res


def stage_evaluate(args, rows):
    import pandas as pd
    import torch
    from peft import PeftModel

    if not os.path.isdir(ADAPTER_DIR):
        sys.exit(f"{ADAPTER_DIR} not found. Run --stage train first.")
    _, _, test = split(rows)
    tok, base = load_tokenizer_and_model(args.base_model)
    tok.padding_side = "left"
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    model.config.use_cache = True

    @torch.no_grad()
    def generate_batch(system, requests, bs=16, max_new=150):
        outs = []
        for i in range(0, len(requests), bs):
            prompts = [build_prompt(tok, system, r) for r in requests[i:i + bs]]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g, skip_special_tokens=True).strip()
                     for g in gen[:, enc["input_ids"].shape[1]:]]
        return outs

    test_req = [r["request"] for r in test]
    test_gold = [r["target"] for r in test]

    with model.disable_adapter():                     # adapter OFF = the original base model
        out_base_short = generate_batch(SYSTEM_SHORT, test_req)
        out_base_schema = generate_batch(SYSTEM_SCHEMA, test_req)
    out_ft = generate_batch(SYSTEM_SHORT, test_req)   # adapter ON

    results = pd.DataFrame({
        "Base (short prompt)": score(out_base_short, test_gold),
        "Base (schema prompt)": score(out_base_schema, test_gold),
        "Fine-tuned (LoRA)": score(out_ft, test_gold),
    }).round(1)
    print(f"\nResults on {len(test)} held-out examples:\n{results}\n")

    fails = []
    for req, g, raw in zip(test_req, test_gold, out_ft):
        p = extract_json(raw)
        wrong = [f for f in FIELDS if p is None or norm_val(f, p.get(f)) != norm_val(f, g[f])]
        if wrong:
            fails.append({"request": req, "gold": g, "pred": raw, "wrong_fields": wrong})
    print(f"{len(fails)} failures out of {len(test_req)}")
    print("Errors per field:", dict(Counter(f for x in fails for f in x["wrong_fields"])))
    for x in fails[:8]:
        print("\nREQUEST  :", x["request"], "\nGOLD     :", json.dumps(x["gold"]),
              "\nPREDICTED:", x["pred"], "\nWRONG    :", x["wrong_fields"])

    os.makedirs(OUT_DIR, exist_ok=True)
    results.to_csv(os.path.join(OUT_DIR, "results_base_vs_finetuned.csv"))
    json.dump(fails, open(os.path.join(OUT_DIR, "failures.json"), "w"), indent=1)
    print(f"\nSaved results and failures to {OUT_DIR}/")


# --------------------------------------------------------------------------- #
# Stage 4 - 7B memory estimate (back-of-envelope, with the working shown)
# --------------------------------------------------------------------------- #
def memory_estimate(P=7e9, layers=32, hidden=4096, mlp=11008, r=16):
    GB = 1e9
    # (a) full fine-tuning, mixed precision with Adam:
    #     fp16 weights 2 + fp16 grads 2 + fp32 master copy 4 + Adam m 4 + Adam v 4 = 16 bytes/param
    full_states = P * (2 + 2 + 4 + 4 + 4) / GB
    act_full = 10                      # assumption: batch 1-4, seq 512, no checkpointing
    # (b) LoRA: frozen fp16 base + trainable adapters on all 7 linear layers
    per_layer = 4 * r * (hidden + hidden) + 3 * r * (hidden + mlp)      # q,k,v,o + gate,up,down
    lora_params = per_layer * layers
    lora_state = lora_params * (4 + 4 + 8) / GB                         # fp32 weights + grads + Adam m,v
    frozen = P * 2 / GB
    act_lora = 4
    # (c) QLoRA: 4-bit NF4 base (+ ~0.2 GB quantisation constants with double quant)
    nf4 = P * 0.5 / GB + 0.2
    act_q = 2.5                        # with gradient checkpointing
    return {
        "full_ft_gb": round(full_states + act_full, 1),
        "full_ft_states_gb": round(full_states, 1),
        "lora_trainable_params_m": round(lora_params / 1e6, 1),
        "lora_gb": round(frozen + lora_state + act_lora, 1),
        "lora_state_gb": round(lora_state, 2),
        "qlora_gb": round(nf4 + lora_state + act_q, 1),
        "qlora_weights_gb": round(nf4, 1),
    }


def stage_memory(_args=None):
    m = memory_estimate()
    print("7B-parameter model, memory needed to train:")
    print(f"  (a) Full FT : states {m['full_ft_states_gb']:.0f} GB + activations ~10 GB = ~{m['full_ft_gb']:.0f} GB")
    print(f"  (b) LoRA    : {m['lora_trainable_params_m']} M trainable params | frozen fp16 weights 14.0 GB "
          f"+ LoRA state {m['lora_state_gb']} GB + activations ~4 GB = ~{m['lora_gb']} GB")
    print(f"  (c) QLoRA   : 4-bit weights {m['qlora_weights_gb']} GB + LoRA state {m['lora_state_gb']} GB "
          f"+ activations ~2.5 GB = ~{m['qlora_gb']} GB")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(m, open(os.path.join(OUT_DIR, "memory_7b.json"), "w"), indent=1)
    return m


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "data", "train", "evaluate", "memory"], default="all")
    ap.add_argument("--no-llm", action="store_true",
                    help="build the dataset from template sentences only (no API key needed)")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--data-path", default=DATA_PATH)
    args = ap.parse_args()

    if args.stage == "memory":
        return stage_memory()
    rows = stage_data(args)
    if args.stage == "data":
        return
    if args.stage in ("all", "train"):
        stage_train(args, rows)
    if args.stage in ("all", "evaluate"):
        stage_evaluate(args, rows)
    if args.stage == "all":
        stage_memory()


if __name__ == "__main__":
    main()
