#!/usr/bin/env python3
"""Vendor-agnostic zero-shot lesion-perception inference for ANY HF VLM via vLLM.

Each model uses its OWN chat template (via AutoProcessor) so the comparison is fair across
vendors. Same fair present/absent prompt as the base eval, same images, row order preserved.

Usage (inside the vLLM container):
  run_vlm_perception.py <model_path> <fairprompt_dataset.jsonl> <media_dir> <out.jsonl> [max_samples]

The dataset is the base-friendly set built by eval_base_perception.py (messages[0]=system,
messages[1]=user with <image>, images=[path], meta). System text is merged into the user turn
so models without a system role still work. Output rows: {"predict","meta"}.
"""
import json, os, sys
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
MAX_PIXELS=262144  # cap to bound vision tokens (avoid OOM)

def main():
    model_path, ds, media_dir, out = sys.argv[1:5]
    max_samples = int(sys.argv[5]) if len(sys.argv) > 5 else None
    rows = [json.loads(l) for l in open(ds) if l.strip()]
    if max_samples:
        rows = rows[:max_samples]

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=4096,
              limit_mm_per_prompt={"image": 1}, enforce_eager=True,
              gpu_memory_utilization=0.9, max_num_seqs=8)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=96, seed=20260613)

    inputs, kept = [], []
    for r in rows:
        sys_t = r["messages"][0]["content"]
        usr_t = r["messages"][1]["content"].replace("<image>", "").strip()
        merged = sys_t + "\n\n" + usr_t          # merge system into user for cross-model safety
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": merged}]}]
        try:
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            # some processors only template plain strings
            prompt = merged
        img = Image.open(os.path.join(media_dir, r["images"][0])).convert("RGB")
        w,h=img.size
        if w*h>MAX_PIXELS:
            import math; sc=math.sqrt(MAX_PIXELS/(w*h)); img=img.resize((max(28,int(w*sc)),max(28,int(h*sc))))
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
        kept.append(r)

    outs = llm.generate(inputs, sp)
    with open(out, "w") as f:
        for r, o in zip(kept, outs):
            f.write(json.dumps({"predict": o.outputs[0].text, "meta": r["meta"]}, ensure_ascii=False) + "\n")
    print(f"{len(kept)} results saved -> {out}")

if __name__ == "__main__":
    main()
