# TIDE Plus — Improvement Plan

> Status: in progress. Author: analysis pass 2026-07-15.
> Scope decision: **extend the existing `tidecv` package in place** (keep TIDE
> compatibility), add new submodules rather than restructuring.
> Priority order chosen: **(1) model output adapters → (2) SAM click benchmark →
> (3) core fixes → (4) measurement/plotting cleanup.**
>
> **Progress (2026-07-15):**
> - ✅ Core fixes #1–#7 landed (AP zero-guard, `has_masks` polygons, confusion-matrix
>   one-to-one matching + FN counting, py<3.12 f-strings, `average_out_summary`
>   per-key averaging, plotting consolidated to `Plotter`, `Plots/Plots` nesting gone).
> - ✅ `tests/` added: 14 passing (core fixes + adapters). Was zero before.
> - ✅ `tidecv/adapters/` built: `base`, `coco`, `yolo` (label-dir + live model),
>   `rfdetr` (detections/json/live). YOLO validated on the real GT folder (167/167
>   annotations aligned, polygon→RLE correct).
> - ✅ `tidecv/interactive/` click benchmark built: `gt_masks` (GT→per-instance
>   masks), `clicker` (deterministic robot-user), `sam_model` (`Sam3LabelingModel`
>   reusing `sam3_labeling`'s `SamBackend`, + `DiskModel` stand-in), `evaluate`
>   (`InteractiveEvaluator` → NoC@IoU / NoF / IoU-curve / per-class), `report`
>   (plots). Runnable CLI at `examples/run_click_benchmark.py`. Harness validated
>   end-to-end on the real GT; `Sam3LabelingModel` import wiring verified against the
>   real repo. 24 tests total, all passing.
> - ⏳ Next: run the real SAM2/SAM3 comparison on the GPU box; remaining fix #8
>   (perf); measurement transparency; repo hygiene (untrack `build/`,
>   `*.egg-info/`, `__pycache__/`).

---

## 0. Where we are

`tide_plus` is a fork of [dbolya/tide](https://github.com/dbolya/tide) that adds JSON
export, multi-model / multi-GT comparison, confusion matrices, and plotting on top of
the original TIDE error-decomposition engine.

The working dataset (`D:\Dateien Auslagerung\tide_ground_truth_adapted\GT.json`) is a
116-image / 167-instance **COCO polygon-segmentation** set of sewer-defect classes
(`BCA, BAB, BBA, BBB, BBC`, DIN EN 13508-2), with parallel YOLO-style `labels/*.txt`.

Two distinct goals sit behind this work:

1. **Compare finished models** (YOLO, RF-DETR, SAM2, SAM3) on the same GT — this is
   what TIDE does today, but the input side (getting each model's output into a common
   format) is missing and the calculation side has bugs.
2. **Compare *interactive* SAM models by click-efficiency** — "which SAM reaches a good
   mask in the fewest clicks." TIDE cannot express this at all; it needs a new harness
   (the standard metric is **NoC@IoU**, Number-of-Clicks to reach a target IoU).

The plan below covers both, plus the fixes that make the existing engine trustworthy.

---

## 1. Model Output Adapters  *(priority 1)*

**Problem.** `datasets.py` can only read COCO/LVIS/Pascal/Cityscapes annotation files.
There is no path from a *model's native output* (YOLO `.txt`, RF-DETR detections, SAM
masks) into the common `Data` object. Every comparison today requires hand-converting
predictions to a COCO results JSON first.

**Goal.** One normalization layer: any supported model → `tidecv.Data`. This is the
"model output understander" — it understands each format and speaks `Data`.

### 1.1 New package: `tidecv/adapters/`

```
tidecv/adapters/
    __init__.py          # registry + public API: load_predictions(model, source, gt, mode)
    base.py              # PredictionAdapter ABC -> Data
    yolo.py              # Ultralytics: .txt label dirs (box + seg polygon) AND live model
    rfdetr.py            # RF-DETR: supervision.Detections / COCO-json / live model
    coco.py              # thin wrapper over existing datasets.COCOResult (canonical form)
    sam.py               # SAM2/SAM3 masks -> Data (prompted-from-GT, see §2 for interactive)
```

Each adapter implements:

```python
class PredictionAdapter(ABC):
    def to_data(self, name: str | None = None) -> Data: ...
```

and returns a `Data` object with detections added via `data.add_detection(image_id,
class_id, score, box, mask)`. Class-id alignment to the GT's category ids is handled in
`base.py` (a `class_map` argument; YOLO indices and COCO category ids rarely match).

### 1.2 YOLO adapter (`yolo.py`)

Two entry points:

- **From label/prediction dirs** — read `labels/*.txt`. Support both:
  - detection: `cls cx cy w h [conf]` (normalized) → abs `[x,y,w,h]` box.
  - segmentation: `cls x1 y1 x2 y2 …` (normalized polygon) → polygon → box via
    `functions.polyToBox`, and RLE via `functions.toRLE` for mask mode.
  Needs image width/height → take from the GT `Data` (match by `file_name`), so the
  adapter takes the GT `Data` as an argument to resolve `image_id` + dimensions.
- **From a live Ultralytics model** — `YOLO("best.pt").predict(images)` → iterate
  `results[i].boxes` / `.masks` → `Data`. Optional (guarded import); the dir-based path
  is the default because it has no heavy dependency.

Note: the existing `draw_mask.py` in the GT folder already parses exactly this YOLO
polygon format — reuse that parsing logic.

### 1.3 RF-DETR adapter (`rfdetr.py`)

RF-DETR (box detector) outputs `supervision.Detections` or a COCO-style results list.
Adapter accepts either and maps to boxes. Guarded import of `rfdetr` for the live path;
otherwise consume a saved detections JSON.

### 1.4 SAM adapter (`sam.py`)

SAM is promptable — it produces a mask *given a prompt*, not a labeled detection set.
For the **static** comparison table, run SAM prompted by each GT box/point and treat the
output mask as that instance's prediction (upper-bound "if you point at the right
object" quality). The **interactive** click-efficiency comparison is §2. Reuse the
`SamBackend` from `sam3_labeling` (see §2.1) for inference here too.

### 1.5 Public API

```python
from tidecv import TIDE, datasets, adapters

gt    = datasets.COCO(r"...\GT.json")
yolo  = adapters.yolo.from_label_dir(r"...\yolo_preds\labels", gt, task="seg")
rfd   = adapters.rfdetr.from_json(r"...\rfdetr_preds.json", gt)

tide = TIDE()
tide.evaluate_multiple_models_on_one_gt(gt, [yolo, rfd], names=["YOLO", "RF-DETR"])
```

**Deliverable:** any of the four model families → `Data` with one call, class-id
alignment handled, no manual COCO-JSON step.

---

## 2. Interactive SAM Click-Efficiency Benchmark  *(priority 2 — the new capability)*

**Problem.** The question "which finetuned SAM needs the fewest clicks for a good mask"
is interactive segmentation, orthogonal to mAP. TIDE has no concept of it.

**Metric (standard in the field — RITM / SimpleClick / SAM eval):**

- **NoC@85 / NoC@90** — mean number of clicks to reach IoU ≥ 85 % / 90 % with the GT
  instance mask.
- **NoF@k** — number of instances that *fail* to reach the target within `k` clicks
  (e.g. `k=20`).
- **IoU-vs-#clicks curve** — mean IoU after 1,2,…,k clicks (the full picture, not just
  the threshold crossing).
- Per-class breakdown (does one SAM handle `BBC` cracks better than another?).

### 2.1 Reuse the existing SAM backend — do NOT rebuild inference

`C:\Code Python\sam3_labeling\src\model\sam_backend.py` already provides a
family-neutral backend that is exactly what the benchmark needs:

```python
backend = SamBackend(model_dir="…/models/sam3", family="auto")  # or models/sam2-v6
backend.load_model()
backend.set_image(pil_image)                    # encodes once, caches embedding
result = backend.predict([Point(x, y, positive=True), ...])   # -> MaskResult(mask, score)
```

- `Point` (`.x, .y, .positive`) and `MaskResult` (`.mask` bool H×W, `.score`) come from
  `sam3_labeling/src/core/annotation.py`.
- Auto-detects SAM2 vs SAM3 from the snapshot's `config.json` `model_type` — so loading
  several finetunes is just a list of `model_dir`s.
- Embedding is cached per image, so iterative clicks on one image are cheap.

Integration options (decide at build time):
- **(a)** `pip install -e` the `sam3_labeling` src as a dependency and import
  `SamBackend`, or
- **(b)** vendor a trimmed copy of `sam_backend.py` + `registry.py` + `annotation.py`
  into `tidecv/interactive/sam_backend/` to keep `tide_plus` self-contained.

Recommendation: **(a)** during development (single source of truth), revisit vendoring
only if `tide_plus` must ship standalone.

### 2.2 New package: `tidecv/interactive/`

```
tidecv/interactive/
    __init__.py
    gt_masks.py          # COCO polygon GT -> per-instance binary masks (pycocotools/cv2)
    clicker.py           # simulate the next click from the current error region
    sam_model.py         # thin protocol wrapping SamBackend: predict(points)->mask
    evaluate.py          # InteractiveEvaluator: run the click loop, collect metrics
    report.py            # NoC tables + IoU-vs-clicks plots
```

### 2.3 The click simulator (`clicker.py`) — the core algorithm

Standard robot-user protocol:

1. **First click:** positive, at the point of the GT mask farthest from its boundary
   (max of the distance transform) — the "most interior" point.
2. **Each subsequent click:** compute `error = XOR(pred_mask, gt_mask)`, split into the
   false-negative region (GT not covered) and false-positive region (pred spilled out).
   Take the **larger** error region, place the click at *its* distance-transform max —
   **positive** if it's a false-negative, **negative** if it's a false-positive.
3. Feed the accumulated click list back to `backend.predict(points)`; SAM's mask input
   from the previous step can be threaded in for models that accept it.
4. Stop at IoU ≥ target or `max_clicks`.

This is deterministic (no randomness) so model A vs model B is a fair comparison.

### 2.4 Evaluator + output

```python
from tidecv.interactive import InteractiveEvaluator, SamModel

ev = InteractiveEvaluator(gt=datasets.COCO(r"...\GT.json"),
                          image_dir=r"...\images",
                          target_iou=0.90, max_clicks=20)

ev.add_model("SAM3",     SamModel(r"...\models\sam3"))
ev.add_model("SAM2-v6",  SamModel(r"...\models\sam2-v6"))
ev.add_model("SAM2-ft",  SamModel(r"...\models\sam2-kanal-ft"))

report = ev.run()          # {model: {NoC@85, NoC@90, NoF@20, iou_curve, per_class}}
ev.save_report(r"...\click_report.json")
ev.plot(r"...\ClickBench")  # bar: NoC@90 per model; line: IoU vs #clicks; per-class NoC
```

**Deliverable:** a ranked table — *SAM-X reaches 90 % IoU in an average of N clicks,
fails on M instances* — plus curves, per class. This directly answers "which SAM is
better" for the labeling workflow.

---

## 3. Core Engine Fixes  *(priority 3 — correctness)*

Concrete bugs found during analysis, with locations:

| # | File / line | Bug | Fix |
|---|-------------|-----|-----|
| 1 | `ap.py:161` `get_mAP` | `sum(aps)/len(aps)` → `ZeroDivisionError` when every class is empty | guard `len==0 → 0.0` |
| 2 | `data.py:106` `has_masks` | Doesn't detect **polygon-list** masks (only RLE dicts); redundant double-`return True` | treat non-empty `list` polygons as masks too; collapse the dead branch |
| 3 | `quantify.py:158` `_compute_confusion_matrix` | No one-to-one matching → one GT counted by many preds; **misses false-negatives** when an image has both preds & GTs (unmatched GTs never added) | reuse the greedy IoU matching already in `TIDEExample`; add unmatched GTs as FN rows in every branch |
| 4 | `tide.py:301,305` `print_summary` | Nested same-quote f-strings need **Python ≥3.12** (PEP 701); crash on 3.8–3.11 | pull sub-expressions into locals / use single quotes inside |
| 5 | `tide.py:134` `average_out_summary` | `break`/`print` on type mismatch silently corrupts the average; averages non-comparable per-class dicts with a uniform divisor | skip non-numeric keys explicitly; per-key counts; document what "Combined Average" means |
| 6 | plotting | Output nests `Plots/Plots/` — both `tide.plot` and `Plotter.plot_a_summary` append `"Plots"` | pick one convention; see §5 |
| 7 | `functions.py:155` | Confusion-matrix plot branch is **dead code** (summary dict never carries `confusion_matrix`) | remove; `Plotter` already does this correctly |
| 8 | perf | `evaluate()` runs 14 full `TIDERun`s per model (error calc only needed on the primary threshold); `__calculate_summary` re-runs on every call | cache; run error decomposition once; only sweep AP across thresholds |

Add a `tests/` suite (currently **zero tests**) covering: AP on a tiny hand-built
example, each error type triggering, confusion-matrix counts, and adapter round-trips.

---

## 4. Measurement / Calculation Improvements  *(priority 4)*

- **Segmentation-first defaults.** For polygon GT, `AUTO` mode currently mis-detects
  masks (bug #2) and silently evaluates boxes. After the fix, default to MASK when the
  GT has segmentation. Make the chosen mode explicit in the summary JSON.
- **mAP definition transparency.** `COCO_THRESHOLDS` was expanded to `0.30–0.95`; only
  `0.50–0.95` feed `mAP 50:95`. Record in the summary exactly which thresholds were
  averaged so numbers are reproducible and comparable to `pycocotools`.
- **Small-N awareness.** 167 instances across 5 classes is small; per-class AP is noisy.
  Report per-class support counts next to each AP, and consider bootstrap CIs on the
  headline mAP so "SAM-A > SAM-B" claims are backed by more than one decimal.
- **Sanity cross-check.** Add an optional `pycocotools` cross-check of `mAP 50:95` on
  one model to confirm the in-house AP matches the reference within tolerance.

---

## 5. Plotting / Reporting Consolidation  *(priority 4)*

- **One plotting path.** There are currently two — `functions.plot` (older, broken
  confusion-matrix branch) and `plotter.Plotter` (newer, correct, loads the separate
  `_confusion_matrices.json`). Keep **`Plotter`**, delete `functions.plot`, point the
  README and `TIDE.plot()` at it, and fix the `Plots/Plots/` nesting (§3 #6).
- **German/English mix.** Comments/titles/`print`s mix German and English
  ("dAP-Anteile", "Alle Plots erstellt"). Pick one for user-facing strings.
- **Single HTML report (optional).** A `report.py` that emits one self-contained page
  (mAP table, error bars, confusion matrices, and — for SAM — the NoC table and
  IoU-vs-clicks curves) makes model comparison a single artifact to share.

---

## 6. Repo Hygiene

- Stop committing build artifacts: `build/`, `tide_plus.egg-info/`, `__pycache__/`,
  `*.pyc` (add to `.gitignore`; they're already tracked and should be removed).
- Pin runtime deps in `pyproject.toml` (numpy, matplotlib, seaborn, pycocotools,
  scikit-learn, appdirs) and make the interactive extras optional
  (`tide_plus[interactive]` → torch/transformers via `sam3_labeling`).
- README: replace the broken `functions.plot` example, document the adapters and the
  click benchmark.

---

## 7. Suggested build order

1. **Adapters** (`tidecv/adapters/`): COCO wrapper → YOLO → RF-DETR → SAM-static.
   Unblocks fair multi-model comparison on the real GT. *(priority 1)*
2. **Core fixes** #1–#3 (AP guard, `has_masks`, confusion matrix) — small, high-value,
   make every number above trustworthy. Land alongside adapters with tests.
3. **Interactive benchmark** (`tidecv/interactive/`), reusing `sam3_labeling`'s
   `SamBackend`. The flagship new capability. *(priority 2)*
4. Remaining fixes #4–#8, measurement transparency, plotting consolidation, hygiene.

Steps 1–2 are independently shippable; step 3 is the headline feature; step 4 is polish.
