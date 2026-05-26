# Task Evidence Validity

Shared EO foundation model representations are not automatically valid task evidence under every observation context. We build a task evidence memory from frozen EO-FM features on the training split and use OT to estimate whether target-observation features can be reliably related to this memory. The resulting validity score is used for diagnostics or optional calibration before downstream prediction.

## What The Memory Stores

The task evidence memory is a saved `.pt` file containing frozen Copernicus-FM features extracted from the downstream task training split. For the first implementation, classification uses pooled image-level features returned by the Copernicus-FM classification wrapper.

Each memory file records:

```python
{
    "features": Tensor[M, D],
    "feature_type": "pooled",
    "dataset_name": str,
    "task": str,
    "model_name": str,
    "created_from": "train_split_only",
    "feature_dim": int,
    "normalization": "l2",
    "metadata": dict,
}
```

Validation and test samples are never used to build this memory.

## Build A Classification Memory

Run from `Copernicus-Bench`:

```bash
python tools/build_task_evidence_memory.py \
  --model-config src/configs/model/copernicusfm_cls.yaml \
  --dataset-config src/configs/dataset/cobench_eurosat_s2.yaml \
  --output output_dir/evidence/cobench_eurosat_s2_copernicusfm.pt \
  --feature-type pooled \
  --memory-size 1024 \
  --max-batches -1 \
  --device cuda
```

For a quick dry run:

```bash
python tools/build_task_evidence_memory.py \
  --dataset-config src/configs/dataset/cobench_eurosat_s2.yaml \
  --output output_dir/evidence/dry_run.pt \
  --max-batches 2 \
  --device cpu
```

## Enable Evidence Mode

Baseline behavior is unchanged by default:

```yaml
use_evidence_memory: false
```

To log evidence diagnostics only:

```bash
TASK_EVIDENCE_MEMORY=output_dir/evidence/cobench_eurosat_s2_copernicusfm.pt \
python src/main.py \
  model=copernicusfm_cls_evidence_score_only \
  dataset=cobench_eurosat_s2 \
  task=classification
```

To apply optional logit scaling:

```bash
TASK_EVIDENCE_MEMORY=output_dir/evidence/cobench_eurosat_s2_copernicusfm.pt \
python src/main.py \
  model=copernicusfm_cls_evidence_logit_scale \
  dataset=cobench_eurosat_s2 \
  task=classification
```

You can also override fields directly:

```bash
python src/main.py \
  model=copernicusfm_cls \
  dataset=cobench_eurosat_s2 \
  task=classification \
  model.use_evidence_memory=true \
  model.memory_path=output_dir/evidence/cobench_eurosat_s2_copernicusfm.pt \
  model.calibration_mode=score_only
```

## Logged Metrics

When evidence mode is enabled, the classification wrapper logs:

- `mean_ot_cost`
- `mean_transport_entropy`
- `mean_validity`
- `low_validity_fraction`

These are logged with the existing Lightning logging style and do not change loss or metrics when `calibration_mode: score_only`.

## Extension Hooks

Segmentation, regression, and change detection configs include default-disabled task evidence fields. The first working implementation is classification-only; later extensions can pool final feature maps or post-pre difference features before the neck or decoder.
