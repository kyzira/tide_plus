"""Interactive SAM click-efficiency benchmark — compare finetuned SAM models by how many
clicks each needs to reach a good mask (NoC@IoU).

Run this in the environment that has CUDA + transformers + the SAM weights (e.g. the
sam3_labeling `.venv`). It reuses sam3_labeling's SamBackend for inference, so point
--sam3-root at that repo checkout.

Example
-------
    python examples/run_click_benchmark.py ^
        --gt      "D:\\Dateien Auslagerung\\tide_ground_truth_adapted\\GT.json" ^
        --images  "D:\\Dateien Auslagerung\\tide_ground_truth_adapted\\images" ^
        --sam3-root "C:\\Code Python\\sam3_labeling" ^
        --model SAM3=C:\\Code Python\\sam3_labeling\\models\\sam3 ^
        --model SAM2-v6=C:\\Code Python\\sam3_labeling\\models\\sam2-v6 ^
        --model SAM2-ft=C:\\Code Python\\sam3_labeling\\models\\sam2-kanal-ft ^
        --out    "results/ClickBench" ^
        --target 0.90 --max-clicks 20

Each --model is NAME=PATH. Add as many finetunes as you like; the family (sam2/sam3) is
auto-detected from each snapshot's config.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tidecv import datasets
from tidecv.interactive import InteractiveEvaluator, Sam3LabelingModel


def parse_model(spec: str):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--model must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="COCO GT json")
    ap.add_argument("--images", required=True, help="directory of images")
    ap.add_argument("--model", action="append", type=parse_model, required=True,
                    metavar="NAME=PATH", help="a SAM snapshot to benchmark (repeatable)")
    ap.add_argument("--sam3-root", default=os.environ.get("SAM3_LABELING_ROOT"),
                    help="path to the sam3_labeling repo (for SamBackend import)")
    ap.add_argument("--out", default="results/ClickBench", help="output directory")
    ap.add_argument("--target", type=float, default=0.90, help="primary IoU target")
    ap.add_argument("--iou-targets", type=float, nargs="+", default=[0.85, 0.90],
                    help="IoU targets to report NoC/NoF for")
    ap.add_argument("--max-clicks", type=int, default=20)
    ap.add_argument("--max-instances", type=int, default=None,
                    help="cap the number of GT instances (for a quick trial run)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--crop-mode", default="full", choices=["full", "crop"])
    args = ap.parse_args()

    gt = datasets.COCO(args.gt)
    print(f"GT: {len(gt.images)} images, {len(gt.annotations)} annotations, "
          f"classes={list(gt.classes.values())}")

    ev = InteractiveEvaluator(
        gt, image_dir=args.images, target_iou=args.target,
        max_clicks=args.max_clicks, iou_targets=tuple(args.iou_targets),
        max_instances=args.max_instances)

    for name, path in args.model:
        ev.add_model(name, Sam3LabelingModel(
            path, name=name, sam3_labeling_root=args.sam3_root,
            device=args.device, dtype=args.dtype, crop_mode=args.crop_mode))
        print(f"  + {name}: {path}")

    ev.run()
    ev.print_report()

    os.makedirs(args.out, exist_ok=True)
    ev.save_report(os.path.join(args.out, "click_report.json"))
    ev.plot(args.out)
    print(f"\nDone. Report + plots in {args.out}")


if __name__ == "__main__":
    main()
