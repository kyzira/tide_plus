# TIDE Plus — Toolbox for Comparing Detection & Segmentation Models

```text
████████╗██╗██████╗ ███████╗
╚══██╔══╝██║██╔══██╗██╔════╝
   ██║   ██║██║  ██║█████╗
   ██║   ██║██║  ██║██╔══╝
   ██║   ██║██████╔╝███████╗
   ╚═╝   ╚═╝╚═════╝ ╚══════╝
```

**TIDE Plus** extends [TIDE (ECCV 2020)](https://github.com/dbolya/tide) into a practical
workbench for **comparing object-detection and instance-segmentation models** on the same
ground truth. It keeps TIDE's error decomposition (why is my mAP not higher?) and adds:

- **Model-output adapters** — drop in YOLO, RF-DETR, SAM, or any COCO-style output and
  have it normalized into one common format automatically ("model output understander").
- **Multi-model / multi-dataset comparison** with JSON export and comparison plots.
- **Normalized confusion matrices** per model.
- **An interactive SAM click-efficiency benchmark** — measure *how many clicks* each
  (finetuned) SAM model needs to produce a good mask (`NoC@IoU`). This is the tool for
  answering *"which SAM is better for my labeling workflow?"*

There are two workflows in this repo, and they answer different questions:

| Workflow | Question it answers | Entry point |
|----------|---------------------|-------------|
| **Static evaluation** | How accurate are these finished predictions vs. GT? (mAP, error types, confusion) | `tidecv.TIDE` |
| **Interactive benchmark** | Which promptable SAM reaches a good mask in the fewest clicks? | `tidecv.interactive.InteractiveEvaluator` |

---

## Table of contents

1. [Installation](#installation)
2. [Concepts: `Data`, GT, predictions](#concepts)
3. [Quick start — evaluate one model](#quick-start)
4. [Model-output adapters (YOLO / RF-DETR / SAM / COCO)](#adapters)
5. [Comparing multiple models](#comparing-models)
6. [What the metrics mean](#metrics)
7. [The interactive SAM click benchmark](#interactive)
8. [Outputs: JSON + plots](#outputs)
9. [Package layout](#layout)
10. [Testing](#testing)
11. [Citation & license](#citation)

---

<a name="installation"></a>
## 1. Installation

```bash
git clone https://github.com/kyzira/tide-plus.git
cd tide-plus
pip install .
```

Core install (above) covers static evaluation and reading model output *files*. Optional
extras are only needed if you want to *run* the models live:

```bash
pip install ".[yolo]"         # run an Ultralytics YOLO model
pip install ".[rfdetr]"       # run RF-DETR live
pip install ".[interactive]"  # torch + transformers for the SAM click benchmark
pip install ".[test]"         # pytest
```

The interactive SAM benchmark additionally reuses the **[`sam3_labeling`](../sam3_labeling)**
project for inference — see [§7](#interactive).

Python ≥ 3.8 is supported.

---

<a name="concepts"></a>
## 2. Concepts: `Data`, GT, predictions

Everything TIDE Plus evaluates is a `tidecv.Data` object — a flat list of annotations,
each with an image id, class id, score, box (`[x, y, w, h]`) and/or mask (COCO RLE or
polygon). You never build `Data` by hand: **loaders** produce a ground-truth `Data`, and
**adapters** produce a predictions `Data`.

- Ground truth comes from `tidecv.datasets` (COCO, LVIS, Pascal VOC, Cityscapes).
- Predictions come from `tidecv.adapters` (YOLO, RF-DETR, SAM, COCO results).

Ground truth and predictions must share the **same class ids** and reference the **same
images**. Adapters handle both alignments for you (class remapping via `class_map`, image
matching by file name).

---

<a name="quick-start"></a>
## 3. Quick start — evaluate one model

```python
from tidecv import TIDE, datasets, adapters

# Ground truth (COCO-style annotations file)
gt = datasets.COCO("GT.json")

# Predictions — here a COCO results json; see §4 for YOLO / RF-DETR / SAM
preds = adapters.coco.from_json("model_results.json")

tide = TIDE()                     # mode defaults to AUTO (mask if both sides have masks)
tide.evaluate(gt, preds, name="MyModel")
tide.print_summary()              # console tables: mAP, error types, precision/recall
tide.print_confusion_matrices()
tide.save_summary("out/summary.json")
tide.plot("out/plots")            # comparison plots + confusion matrices
```

Force box or mask evaluation explicitly:

```python
tide = TIDE(mode=TIDE.MASK)       # or TIDE.BOX, or TIDE.AUTO (default)
```

---

<a name="adapters"></a>
## 4. Model-output adapters

`tidecv.adapters` turns each model's native output into a predictions `Data`. All adapters
align image ids by **file name** (matched against the GT) and can remap class indices with
a `class_map` (`{model_class: gt_category_id}`; default is identity). Polygon masks are
converted to COCO RLE so both box- and mask-mode evaluation work.

### YOLO (Ultralytics)

From a directory of YOLO `.txt` label files (detection `cls cx cy w h [conf]` **or**
segmentation `cls x1 y1 … [conf]`, all normalized):

```python
yolo = adapters.yolo.from_label_dir(
    label_dir="preds/labels",
    gt=gt,
    image_dir="images",       # needed to denormalize coordinates
    task="seg",               # "auto" | "detect" | "seg"
    class_map=None,           # e.g. {0: 3, 1: 1} if YOLO indices != GT ids
    name="YOLOv8",
)
```

Or run a live model (needs `pip install ".[yolo]"`):

```python
yolo = adapters.yolo.from_model("best.pt", images="images", gt=gt, name="YOLOv8")
```

### RF-DETR

```python
# from a saved COCO-style results json
rfd = adapters.rfdetr.from_json("rfdetr_results.json")

# from supervision.Detections (one image at a time)
data = adapters.rfdetr.from_detections(detections, image_id=7, gt=gt)

# live model (needs pip install ".[rfdetr]")
rfd = adapters.rfdetr.from_model("rfdetr.pth", images="images", gt=gt)
```

### COCO results (canonical form)

```python
preds = adapters.coco.from_json("results.json")   # any COCO-format detector output
```

### SAM (static, prompted from GT)

For a static mask-quality comparison, SAM is prompted by each GT instance and its output
mask is scored against that instance. (For the *interactive* click-efficiency comparison,
see [§7](#interactive) — that's the more informative SAM benchmark.)

---

<a name="comparing-models"></a>
## 5. Comparing multiple models

### Several models, one ground truth

```python
gt = datasets.COCO("GT.json")
yolo = adapters.yolo.from_label_dir("yolo/labels", gt, image_dir="images", task="seg")
rfd  = adapters.coco.from_json("rfdetr_results.json")

tide = TIDE()
tide.evaluate_multiple_models_on_one_gt(gt, [yolo, rfd], names=["YOLO", "RF-DETR"])
tide.print_summary()
tide.plot("out/Plots")
```

Produces mAP bars, threshold-AP curves, error breakdowns, precision/recall, per-class
charts, and one confusion matrix per model.

### One model, several ground truths (splits / datasets)

```python
tide = TIDE()
tide.evaluate_model_on_multiple_gt(
    gt_list=[datasets.COCO("val.json"), datasets.COCO("extra.json")],
    preds_list=[preds_val, preds_extra],
    name="YOLO",
)
tide.average_out_summary()            # adds a "Combined Average" entry
```

---

<a name="metrics"></a>
## 6. What the metrics mean

- **mAP 50:95** — mean AP averaged over IoU thresholds 0.50…0.95 (COCO standard).
- **Threshold AP @** — AP at each IoU threshold in `TIDE.COCO_THRESHOLDS`.
- **Precision / Recall** — overall and per size bucket (Small / Medium / Large).
- **Main Errors (dAP)** — TIDE's error decomposition: how much mAP you'd recover by fixing
  each error type. Fixing them one at a time isolates the contribution of each:
  - `Cls` — right box, wrong class.
  - `Loc` — right class, box not tight enough (`bg_thresh ≤ IoU < pos_thresh`).
  - `Both` — wrong class *and* wrong location.
  - `Dupe` — correct, but a better detection already claimed that GT.
  - `Bkg` — detected background (IoU below `bg_thresh`).
  - `Miss` — a GT no detection covered.
- **Special Errors (dAP)** — `FalsePos` (AP recoverable with perfect precision) and
  `FalseNeg` (AP recoverable with perfect recall).
- **Confusion matrix** — row-normalized, one-to-one matched (each GT claimed by its most
  confident detection), with a `background` row/column for false positives and false
  negatives.

Thresholds are configurable:

```python
tide = TIDE(pos_threshold=0.5, background_threshold=0.1, mode=TIDE.AUTO)
```

---

<a name="interactive"></a>
## 7. The interactive SAM click benchmark

**Goal:** compare promptable segmentation models (SAM 2, SAM 3, your finetunes) by *how
many clicks* each needs to reproduce a GT instance mask. The metric is **NoC@IoU** — the
mean Number of Clicks to reach a target IoU (e.g. 85 %, 90 %).

### How it works

For every GT instance, a deterministic **robot user** simulates clicking:

1. First click: positive, at the most-interior point of the GT mask.
2. Each later click: look at the error between the model's current mask and the GT; take
   the larger error region and click its most-interior point — **positive** if the model
   *missed* GT there, **negative** if it *spilled* outside GT.
3. Stop when IoU ≥ target or `max_clicks` is reached.

Because it's deterministic, model-A-vs-model-B is a fair comparison. Reported metrics:
**NoC@85 / NoC@90**, **NoF** (instances that never reach the target), an **IoU-vs-#clicks
curve**, and a **per-class** breakdown.

### Inference backend

Real SAM inference reuses the family-neutral `SamBackend` from the **`sam3_labeling`**
project (it auto-detects SAM 2 vs SAM 3 from each snapshot's `config.json`, so loading a
list of finetunes just works). Point TIDE Plus at that checkout:

```bash
set SAM3_LABELING_ROOT=C:\Code Python\sam3_labeling   # or pass sam3_labeling_root=...
```

### Run it from Python

```python
from tidecv import datasets
from tidecv.interactive import InteractiveEvaluator, Sam3LabelingModel

gt = datasets.COCO(r"GT.json")
ev = InteractiveEvaluator(gt, image_dir=r"images", target_iou=0.90, max_clicks=20,
                          iou_targets=(0.85, 0.90))

ev.add_model("SAM3",    Sam3LabelingModel(r"...\models\sam3"))
ev.add_model("SAM2-v6", Sam3LabelingModel(r"...\models\sam2-v6"))
ev.add_model("SAM2-ft", Sam3LabelingModel(r"...\models\sam2-kanal-ft"))

ev.run()
ev.print_report()
ev.save_report(r"out\click_report.json")
ev.plot(r"out\ClickBench")          # NoC bars, IoU-vs-clicks curves, per-class NoC
```

### Run it from the command line

```powershell
python examples/run_click_benchmark.py `
  --gt      "GT.json" `
  --images  "images" `
  --sam3-root "C:\Code Python\sam3_labeling" `
  --model SAM3=C:\Code Python\sam3_labeling\models\sam3 `
  --model SAM2-v6=C:\Code Python\sam3_labeling\models\sam2-v6 `
  --out     "results/ClickBench" `
  --target 0.90 --max-clicks 20
```

Each `--model NAME=PATH` is repeatable. Add `--max-instances 20` for a quick trial run.
Run this in the environment that has CUDA + `transformers` + the SAM weights (e.g. the
`sam3_labeling` `.venv`).

### Testing the harness without a GPU

`DiskModel` is a dependency-free stand-in that implements the same interface, so the whole
click loop and metrics can run (and be unit-tested) without any model:

```python
from tidecv.interactive import InteractiveEvaluator, DiskModel
ev.add_model("disk", DiskModel(radius=25))
```

---

<a name="outputs"></a>
## 8. Outputs

**Static evaluation** (`tide.save_summary`, `tide.plot`):

- `summary.json` — all metrics per model.
- `summary_confusion_matrices.json` — labels + normalized matrices.
- Plots: `mAP_comparison.png`, `AP_thresholds_comparison.png`,
  `Main_Errors_comparison.png`, `Special_Errors_comparison.png`,
  `Precision_comparison.png`, `Recall_comparison.png`, per-class charts, and one
  `<model>_confusion_matrix.png` each.

To re-plot from saved summaries later:

```python
from tidecv.plotter import Plotter
Plotter.plot_a_summary("out_dir", "out/summary.json")
```

**Interactive benchmark** (`ev.save_report`, `ev.plot`):

- `click_report.json` — `NoC`, `NoF`, `iou_curve`, `mean_iou_final`, `per_class` per model.
- Plots: `NoC_comparison.png`, `IoU_vs_clicks.png`, `per_class_NoC.png`.

---

<a name="layout"></a>
## 9. Package layout

```text
tidecv/
  data.py            Data container (annotations, images, classes)
  datasets.py        GT loaders: COCO, LVIS, Pascal, Cityscapes
  ap.py              average-precision computation
  quantify.py        the TIDE engine (matching, errors, confusion matrix)
  tide.py            TIDE facade: evaluate / summary / plot
  functions.py       helpers (RLE, polygons, tables, plotting entry)
  plotter.py         the maintained comparison-plot implementation
  errors/            TIDE error type definitions
  adapters/          model-output understanders
    base.py            image/class alignment
    coco.py  yolo.py  rfdetr.py
  interactive/       SAM click-efficiency benchmark
    gt_masks.py        GT -> per-instance binary masks
    clicker.py         deterministic robot-user
    sam_model.py       Sam3LabelingModel (real) + DiskModel (test)
    evaluate.py        InteractiveEvaluator (NoC@IoU, NoF, curves, per-class)
    report.py          benchmark plots
examples/
  coco_instance_segmentation.ipynb   static-evaluation walkthrough
  run_click_benchmark.py             interactive benchmark CLI
docs/
  IMPROVEMENT_PLAN.md                roadmap & design notes
tests/                               pytest suite
```

---

<a name="testing"></a>
## 10. Testing

```bash
pip install ".[test]"
pytest -q
```

The suite covers the AP/confusion-matrix engine, the mask/polygon handling, every adapter,
and the interactive click loop + metrics (using `DiskModel`, so no GPU is required).

---

<a name="citation"></a>
## 11. Citation & license

TIDE Plus is built on TIDE and distributed under the same (MIT) license.

```bibtex
@inproceedings{tide-eccv2020,
  author    = {Daniel Bolya and Sean Foley and James Hays and Judy Hoffman},
  title     = {TIDE: A General Toolbox for Identifying Object Detection Errors},
  booktitle = {ECCV},
  year      = {2020},
}
```
