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
        matrices = {}

        # Haupt-Summaries
        for path in self.summary_files:
            with open(path, "r") as f:
                data = json.load(f)
            model_name = os.path.splitext(os.path.basename(path))[0]
            summaries[model_name] = data

            # Prüfe, ob es eine zugehörige *_confusion_matrices.json Datei gibt
            cm_path = path.replace(".json", "_confusion_matrices.json")
            if os.path.exists(cm_path):
                with open(cm_path, "r") as f:
                    matrices[model_name] = json.load(f)

        # Füge Confusion Matrices zu passenden Runs hinzu
        for model_name, cm_data in matrices.items():
            if model_name not in summaries:
                continue
            for run_name, run_data in summaries[model_name].items():
                if run_name in cm_data:
                    run_data["confusion_matrix"] = cm_data[run_name]["matrix"]
                    run_data["class_labels"] = cm_data[run_name]["labels"]

        return summaries

    @staticmethod
    def _plot_single(data_dict, out_dir):
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
            plt.savefig(os.path.join(out_dir, fig_name), dpi=200)
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

            min_val = 100
            max_val = 0
            
            for model, tvals in threshold_values.items():
                for value in tvals.keys():
                    min_val = min(min_val, int(value))
                    max_val = max(max_val, int(value))


            plt.figure(figsize=(10, 6))
            for model, tvals in threshold_values.items():
                plt.plot(tvals.keys(), tvals.values(), marker="o", label=model)
            plt.title(f"AP over Thresholds ({min_val}–{max_val})")
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

        # --- Precision (Größen-basiert) ---
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

        # --- Recall (Größen-basiert) ---
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

        # --- Confusion Matrices ---
        for model, data in data_dict.items():
            if "confusion_matrix" in data and "class_labels" in data:
                cm = np.array(data["confusion_matrix"])
                labels = data["class_labels"]

                plt.figure(figsize=(20, 18))
                sns.heatmap(
                    cm,
                    annot=True,               # Zahlen anzeigen
                    fmt=".2f",                # auf 2 Nachkommastellen runden
                    cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels,
                    cbar=True
                )
                plt.title(f"Confusion Matrix - {model}")
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.tight_layout()
                save_plot(f"{model}_confusion_matrix.png")


        # --- Per-Class Statistiken (Precision / Recall / AP / mAP50 / mAP50:95) ---
        per_class_metrics = [
            "Per-Class Precision",
            "Per-Class Recall",
            "Per-Class AP",
            "Per-Class mAP 50",
            "Per-Class mAP 50:95",
        ]

        for metric_name in per_class_metrics:
            # Sammle alle Klassen, die in mindestens einem Modell vorkommen
            all_classes = sorted({
                cls
                for data in data_dict.values()
                if metric_name in data
                for cls in data[metric_name].keys()
            })
            if not all_classes:
                continue

            plt.figure(figsize=(max(14, len(all_classes) * 0.5), 6))
            x = np.arange(len(all_classes))
            width = 0.8 / len(data_dict)

            for i, (model, model_data) in enumerate(data_dict.items()):
                if metric_name not in model_data:
                    continue
                metric_data = model_data[metric_name]
                values = [metric_data.get(cls, 0.0) for cls in all_classes]
                plt.bar(x + i * width, values, width, label=model, alpha=0.8)

            plt.xticks(x + width * len(data_dict) / 2, all_classes, rotation=45, ha='right')
            plt.ylabel(metric_name.split()[-1])
            plt.title(f"{metric_name} über alle Modelle")
            plt.legend()
            plt.tight_layout()
            safe_name = (
                metric_name.replace(" ", "_")
                .replace(":", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .lower()
            )
            plt.savefig(os.path.join(out_dir, f"{safe_name}_comparison.png"), dpi=200)
            plt.close()

    @staticmethod
    def plot_a_summary(output_dir, summary_path):
        """Erstellt einen Gesamtplot über alle Modelle."""
        out_path = os.path.join(output_dir, "Plots")
        os.makedirs(out_path, exist_ok=True)

        summary = dict()
        with open(summary_path, "r") as f:
            summary = json.load(f)

        Plotter._plot_single(summary, out_path)




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

        print("Alle Plots erstellt.")