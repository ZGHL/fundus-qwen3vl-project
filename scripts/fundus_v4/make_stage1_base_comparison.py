#!/usr/bin/env python3
"""Create a concise same-sample comparison of base Qwen3-VL and Adapter 1."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path('/workspace/fundus-qwen3vl-project')
BASE=Path('/workspace/LLaMA-Factory/saves/qwen3-vl-8b-fundus/lora/stage1_eval/base_model')
OUT=ROOT/'reports/stage1_adapter1_vs_base_model_20260609.md'

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def pct(x): return f'{100*x:.2f}%'

def table(split, adapter, base):
    lines=[f'## {split}', '', '| Model | Macro F1 | Recall | Specificity | Balanced Acc | JSON Parse | Target Consistency |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in [('Qwen3-VL-8B base',base),('Adapter 1: stage1_en_cot',adapter)]:
        a=m['main4_macro']; lines.append(f'| {name} | {pct(a["f1"])} | {pct(a["recall"])} | {pct(a["specificity"])} | {pct(a["balanced_accuracy"])} | {pct(m["json_parse_success"])} | {pct(m["target_consistency"])} |')
    lines += ['', '| Lesion | Base F1 | Adapter 1 F1 | Delta | Base Recall | Adapter Recall | Base Specificity | Adapter Specificity |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for lesion in ('MA','HE','EX','SE'):
        b=base['by_lesion'][lesion]; a=adapter['by_lesion'][lesion]
        lines.append(f'| {lesion} | {pct(b["f1"])} | {pct(a["f1"])} | {pct(a["f1"]-b["f1"])} | {pct(b["recall"])} | {pct(a["recall"])} | {pct(b["specificity"])} | {pct(a["specificity"])} |')
    return lines

dev_a=load(ROOT/'reports/metrics/stage1_en_cot_gold_dev_metrics.json')
test_a=load(ROOT/'reports/metrics/stage1_en_cot_gold_test_metrics.json')
dev_b=load(BASE/'gold_dev/stage1_metrics.json')
test_b=load(BASE/'gold_test/stage1_metrics.json')
lines=['# Stage1 Adapter 1 vs Qwen3-VL Base Model', '', 'Same prompt, image preprocessing, decoding settings, scorer, and held-out rows are used for each comparison.', '', '- Adapter 1: `saves/qwen3-vl-8b-fundus/lora/stage1_en_cot`', '- Base model: `models/Qwen3-VL-8B-Instruct` without an adapter', '- Selection policy: this report does not use Gold-test to select a model.', '']
lines += table('Gold-dev (596 rows)',dev_a,dev_b)+['']+table('Gold-test (900 rows)',test_a,test_b)
macro_delta=test_a['main4_macro']['f1']-test_b['main4_macro']['f1']
lines += ['', '## Conclusion', '', f'- Adapter 1 Gold-test Macro F1 improvement over base: **{pct(macro_delta)}**.', '- Adapter 1 remains the default Stage2 starting point unless a calibration candidate passes the committed preservation guardrails.', '']
OUT.write_text('\n'.join(lines),encoding='utf-8')
print(OUT)
