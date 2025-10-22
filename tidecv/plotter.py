import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    def __init__(self, summary_files: list[str], output_dir: str):
        self.summary_files = summary_files
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _load_summaries(self):
        summaries = {}
        for path in self.summary_files:
            with open(path, "r") as f:
                data = json.load(f)
            model_name = os.path.splitext(os.path.basename(path))[0]
            summaries[model_name] = data
        return summaries

    def _plot_single(self, data_dict, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        models = []
        map_values = {}
        threshold_values = {}
        main_errors = {}
        special_errors = {}
        precision_values = {}
        recall_values = {}

        for run_name, data in data_dict.items():
            models.append(run_name)
            if "mAP 50:95" in data:
                map_values[run_name] = float(data["mAP 50:95"])
            if "Threshold AP @" in data:
                threshold_values[run_name] = {str(k): float(v) for k, v in data["Threshold AP @"].items()}
            if "Main Errors" in data:
                main_errors[run_name] = {str(k): float(v) for k, v in data["Main Errors"].items()}
            if "Special Errors" in data:
                special_errors[run_name] = {str(k): float(v) for k, v in data["Special Errors"].items()}
            if "Precision" in data:
                precision_values[run_name] = {k: float(v) for k, v in data["Precision"].items()}
            if "Recall" in data:
                recall_values[run_name] = {k: float(v) for k, v in data["Recall"].items()}

        def save_plot(fig_name):
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fig_name))
            plt.close()


		# --- mAP Comparison ---
        if map_values:
            plt.figure(figsize=(10, 6))
            plt.bar(map_values.keys(), map_values.values(), color="steelblue")
            plt.title("mAP 50:95 Comparison")
            plt.ylabel("mAP 50:95")
            plt.xticks(rotation=45, ha='right')
            save_plot("mAP_comparison.png")

        # --- Threshold AP ---
        if threshold_values:
            plt.figure(figsize=(10, 6))
            for model, tvals in threshold_values.items():
                plt.plot(tvals.keys(), tvals.values(), marker="o", label=model)
            plt.title("AP over Thresholds (50–95)")
            plt.xlabel("Threshold")
            plt.ylabel("AP")
            plt.legend()
            save_plot("AP_thresholds_comparison.png")

        # --- Main Errors ---
        if main_errors:
            plt.figure(figsize=(12, 7))
            error_types = sorted({err for v in main_errors.values() for err in v.keys()})
            x = np.arange(len(error_types))
            width = 0.8 / len(models)
            for i, model in enumerate(models):
                y = [main_errors[model].get(err, 0.0) for err in error_types]
                plt.bar(x + i * width, y, width, label=model, alpha=0.8)
            plt.xticks(x + width * (len(models) / 2), error_types, rotation=45, ha='right')
            plt.title("Main Errors (dAP)")
            plt.ylabel("dAP")
            plt.legend()
            save_plot("Main_Errors_comparison.png")

        # --- Special Errors ---
        if special_errors:
            plt.figure(figsize=(12, 7))
            error_types = sorted({err for v in special_errors.values() for err in v.keys()})
            x = np.arange(len(error_types))
            width = 0.8 / len(models)
            for i, model in enumerate(models):
                y = [special_errors[model].get(err, 0.0) for err in error_types]
                plt.bar(x + i * width, y, width, label=model, alpha=0.8)
            plt.xticks(x + width * (len(models) / 2), error_types, rotation=45, ha='right')
            plt.title("Special Errors (dAP)")
            plt.ylabel("dAP")
            plt.legend()
            save_plot("Special_Errors_comparison.png")

        # --- Precision (Modelle unten, Kategorien als Gruppen) ---
        if precision_values:
            plt.figure(figsize=(12, 7))
            ordered_keys = ["AP (Small)", "AP (Medium)", "AP (Large)", "Average"]
            width = 0.15
            x = np.arange(len(models))
            for i, cat in enumerate(ordered_keys):
                y = [precision_values[m].get(cat, 0.0) for m in models]
                plt.bar(x + i * width, y, width, label=cat)
            plt.xticks(x + width * (len(ordered_keys) / 2), models, rotation=45, ha='right')
            plt.title("Precision Overview (Average / Size-based)")
            plt.ylabel("Precision (%)")
            plt.legend()
            save_plot("Precision_comparison.png")

        # --- Recall (Modelle unten, Kategorien als Gruppen) ---
        if recall_values:
            plt.figure(figsize=(12, 7))
            ordered_keys = ["AR (Small)", "AR (Medium)", "AR (Large)", "Average"]
            width = 0.15
            x = np.arange(len(models))
            for i, cat in enumerate(ordered_keys):
                y = [recall_values[m].get(cat, 0.0) for m in models]
                plt.bar(x + i * width, y, width, label=cat)
            plt.xticks(x + width * (len(ordered_keys) / 2), models, rotation=45, ha='right')
            plt.title("Recall Overview (Average / Size-based)")
            plt.ylabel("Recall (%)")
            plt.legend()
            save_plot("Recall_comparison.png")

    def run(self):
        summaries = self._load_summaries()
        gt_groups = {}

        for model_name, summary in summaries.items():
            for key, val in summary.items():
                if key == "Combined Average":
                    gt_groups.setdefault("Combined Average", {})[model_name] = val
                else:
                    gt_id = key.split("_GT_num_")[-1]
                    gt_groups.setdefault(gt_id, {})[model_name] = val

        for gt_id, data in gt_groups.items():
            gt_dir = os.path.join(self.output_dir, f"GT_{gt_id}" if gt_id != "Combined Average" else "Combined_Average")
            self._plot_single(data, gt_dir)

        print("All Plots created!")
