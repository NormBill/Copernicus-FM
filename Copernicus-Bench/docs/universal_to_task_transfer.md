# Universal-to-Task Transfer Interface

EO foundation models learn universal embeddings, but downstream tasks still use them through task-specific predictors trained with direct task losses. This leaves the application of universal representations largely implicit. We introduce a universal-to-task transfer interface that assigns encoder features to task evidence structures before prediction.

The module addresses the universal-to-task application gap in EO foundation models. The EO-FM encoder is universal, but downstream tasks apply its embeddings through task-specific predictors and direct task losses. The transfer interface introduces task evidence atoms and OT-based evidence routing before prediction, making the use of universal embeddings more explicit.

## What Changes

The default classification path remains unchanged when `model.use_transfer_interface=false`.

When enabled, the classification path becomes:

```text
input
-> frozen Copernicus-FM encoder
-> universal feature
-> task evidence atoms
-> local OT evidence routing
-> evidence-calibrated feature
-> task-specific classifier
-> task loss
```

The router output is an assignment-informed `routed_evidence` representation. The scalar diagnostics are secondary transfer interface statistics.

## What It Is Not

This interface is not a detector for whether an input is outside the training data distribution. It does not train on validation or test correctness, and it does not use the router statistics as the main claim. Its purpose is to make the application of universal EO embeddings to a downstream task explicit before prediction.

## Difference From Global Memory Matching

The old validity path used a pooled image feature against a flat memory and produced scalar diagnostics. That remains available only as `use_global_ot_ablation`.

The new interface uses task-structured atoms. For classification, atoms are labelled class evidence from the train split. Routing is local and class-conditioned: preliminary logits select a class, nearest positive and competing atoms are measured, and the selected positive atoms produce routed evidence for feature calibration.

## Memory Construction

Build memory from the training split only:

```bash
python tools/build_task_evidence_memory.py \
  --model-config src/configs/model/copernicusfm_cls.yaml \
  --dataset-config src/configs/dataset/cobench_eurosat_s2.yaml \
  --output outputs/evidence/cobench_eurosat_s2_transfer.pt \
  --feature-type pooled \
  --task-type classification \
  --memory-size 2048 \
  --atoms-per-class 128 \
  --max-batches -1 \
  --device cuda
```

The saved `.pt` file contains:

```python
{
    "features": Tensor[M, D],
    "labels": Tensor[M],
    "atom_types": ["class_evidence", ...],
    "task_type": "classification",
    "dataset_name": "...",
    "model_name": "copernicusfm",
    "feature_type": "pooled",
    "created_from": "train_split_only",
    "num_atoms": M,
    "feature_dim": D,
    "metadata": {...},
}
```

Validation and test samples must never be used to build this memory.

## Enable The Interface

Run the original baseline:

```bash
python src/main.py model=copernicusfm_cls dataset=cobench_eurosat_s2 task=classification
```

Run with task evidence routing and transfer calibration:

```bash
python src/main.py \
  model=copernicusfm_cls \
  dataset=cobench_eurosat_s2 \
  task=classification \
  model.use_transfer_interface=true \
  model.task_evidence_memory_path=outputs/evidence/cobench_eurosat_s2_transfer.pt \
  model.evidence_router_mode=class_conditional \
  model.transfer_calibrator_mode=gated_residual
```

The encoder is frozen by default through `model.freeze_backbone=true`. The classifier head and lightweight transfer calibrator are optimized with the standard downstream task loss.

## Ablations

Use these only for comparisons:

```yaml
model.use_global_ot_ablation: true
model.use_nearest_prototype_ablation: true
model.use_adapter_only_ablation: true
```

The main method is:

```text
universal feature
-> class/task evidence atoms
-> local OT evidence routing
-> evidence-calibrated feature
-> task-specific predictor
```

## Router Statistics

When enabled, the wrapper logs transfer interface statistics:

```text
mean_d_pos
mean_d_neg
mean_evidence_margin
mean_assignment_entropy
mean_gate_value
task_evidence_usage_class_<id>
```

These diagnostics describe evidence assignment and task evidence usage. Main evaluation should use the existing downstream classification metrics such as accuracy or mAP, depending on the dataset.

## Future Hooks

Segmentation can build class-wise patch or region evidence atoms before UPerNet. Regression can build value-bin evidence atoms before the regression head. Change detection can build transition evidence atoms from pre/post difference features before the decoder.
