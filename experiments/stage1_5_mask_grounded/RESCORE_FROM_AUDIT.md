# Plan A — re-score Stage-2 predictions: FREE vs FROM-AUDIT (no retraining)

## Why
The first Stage-2 sweep let the model **freely generate `dr_tier`**. Result: faithfulness
to its own audit was only **0.43** (and it *dropped* as training progressed — ckpt-120 0.66 →
ckpt-360 0.32), ~**25% invalid outputs** (mostly 512-token truncation that killed the final
JSON), and the **Mild** tier collapsed to 0 recall.

All three are symptoms of one cause: free tier generation drifts away from the transparent
`presence → tier` map the data was built on. Plan A fixes them at the scoring layer, on the
**same predictions**, by computing the tier from the model's lesion audit:

`predicted tier = fitted_map[ pattern(model.lesions_present) ]`

- Faithfulness becomes **1.000 by construction** (tier provably follows the stated audit).
- The `[Lesion Audit]` block is emitted **before** the JSON, so truncated outputs that lost
  their JSON are recovered from the audit lines → invalid-tier coverage should jump toward 100%.
- Clinical accuracy then reflects **audit quality** (Stage-1.5 v3 perception), bounded by the
  faithful ceiling 0.688. Gold-audit upper bound (if audit == GT presence): referable
  sens 0.982 / spec 0.886.

## Run on the VM (uses the existing sweep predictions — nothing retrained)

```bash
cd /workspace/stage1_5_experiment   # or wherever this bundle lives on the VM
TEST=data/stage2_grade_test_sft.jsonl
DIST=data/stage2_grade_distribution.json

# point this at the dir holding one *.jsonl of vLLM predictions per checkpoint
PREDS=<dir-with-checkpoint-prediction-jsonls>

python3 scripts/rescore_stage2.py "$TEST" "$PREDS" "$DIST" reports/STAGE2_RESCORE_FROM_AUDIT.md
```

`rescore_stage2.py` emits, for every checkpoint, both `· free` and `· audit` rows side by side
(valid / QWK / MacroF1 / MAE / RefSens / RefSpec / SevRecall / Faithful / Fab / Abstain) plus an
audit-source recovery table (how many predictions were rescued from the audit after JSON
truncation). Single checkpoint, one mode:

```bash
python3 scripts/score_stage2.py "$TEST" <one_pred.jsonl> out.md --from-audit --dist "$DIST"
```

Predictions align by **row order** to the test file (300 rows, 60/tier, Adapter1-unseen).
If the VM no longer has the raw prediction jsonls, re-run vLLM inference for the candidate
checkpoints first (temp 0, top-p 1) and keep the per-row `predict` field.

## What to look for
- Does FROM-AUDIT lift **valid-tier** toward 1.0 and recover Mild? (Mild = `MA`-only pattern.)
- Does referable **sens/spec** improve vs free, and how close to the gold-audit bound (0.982/0.886)?
- Faithful is 1.000 by construction — the open question is whether the **audit itself** is good
  enough (it is bounded by Stage-1.5 v3, whose MA on aptos-domain is the weak link behind Mild).

If FROM-AUDIT clearly wins, the next training run should **stop training the model to emit a
tier at all** — train only the audit + decision-path, compute the tier in code. That makes the
whole framework faithful by construction and removes the truncation/format failure mode.
