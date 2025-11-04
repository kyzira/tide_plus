from .data import Data
from .errors.main_errors import *
from . import functions as f
from collections import OrderedDict
from .quantify import TIDERun
import os
from .errors.qualifiers import Qualifier, AREA
import numpy as np

class TIDE:
	"""
	████████╗██╗██████╗ ███████╗
	╚══██╔══╝██║██╔══██╗██╔════╝
	   ██║   ██║██║  ██║█████╗  
	   ██║   ██║██║  ██║██╔══╝  
	   ██║   ██║██████╔╝███████╗
	   ╚═╝   ╚═╝╚═════╝ ╚══════╝
   """

	

	# The modes of evaluation
	BOX  = 'bbox'
	MASK = 'mask'
	AUTO = 'auto'

	def __init__(self, pos_threshold:float=0.5, background_threshold:float=0.1, mode:str=AUTO):
		self.pos_thresh = pos_threshold
		self.bg_thresh  = background_threshold
		self.mode       = mode

		self._error_types = [ClassError, BoxError, OtherError, DuplicateError, BackgroundError, MissedError]
		self._special_error_types = [FalsePositiveError, FalseNegativeError]

		self.pos_thresh_int = int(self.pos_thresh * 100)

		self.summary = {}
		self.runs = {}
		self.run_thresholds = {}
		self.run_main_errors = {}
		self.run_special_errors = {}

		self.qualifiers = OrderedDict()

		self.COCO_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

	def __calculate_summary(self):
		main_errors = self.get_main_errors()
		special_errors = self.get_special_errors()

		for run_name, run in self.runs.items():
			run_summary = {}
			thresh_runs = self.run_thresholds.get(run_name, [])
			aps = [trun.ap for trun in thresh_runs if 0.5 <= trun.pos_thresh <= 0.95]
			run_summary["mAP 50:95"] = sum(aps) / len(aps) if aps else 0.0

			run_summary["Threshold AP @"] = {
				str(int(trun.pos_thresh * 100)): round(trun.ap, 2)
				for trun in thresh_runs
			}

			run_summary["Precision"] = {"Average": round(run.ap_data.get_precision() * 100, 2)}
			run_summary["Recall"] = {"Average": round(run.ap_data.get_recall() * 100, 2)}

			size_quals = {"Small": AREA[0], "Medium": AREA[1], "Large": AREA[2]}
			for size_name, qual in size_quals.items():
				ap_data = run.apply_qualifier(qual)
				run_summary["Precision"][f"AP ({size_name})"] = round(ap_data.get_mAP(), 2)
				run_summary["Recall"][f"AR ({size_name})"] = round(ap_data.get_recall() * 100, 2)

			run_summary["Main Errors"] = {
				(err.short_name if hasattr(err, "short_name") else str(err)): 
				round(main_errors[run_name].get(err.short_name if hasattr(err, "short_name") else str(err), 0.0), 2)
				for err in self._error_types
			}
			run_summary["Special Errors"] = {
				(err.short_name if hasattr(err, "short_name") else str(err)): 
				round(special_errors[run_name].get(err.short_name if hasattr(err, "short_name") else str(err), 0.0), 2)
				for err in self._special_error_types
			}

			run_summary["Per-Class Precision"] = run.precision_per_class
			run_summary["Per-Class Recall"] = run.recall_per_class
			run_summary["Per-Class AP"] = run.ap_per_class

			# --- Per-class mAP @50 und mAP @50–95 ---
			per_class_map_50 = {}
			per_class_map_50_95 = {}
			for cls_name in run.gt.classes.values():
				aps_at_thresh = [
					(trun.pos_thresh, trun.ap_per_class[cls_name])
					for trun in thresh_runs
					if hasattr(trun, "ap_per_class") and cls_name in trun.ap_per_class
				]
				ap_50 = next((ap for t, ap in aps_at_thresh if abs(t - 0.5) < 1e-6), 0.0)
				aps_range = [ap for t, ap in aps_at_thresh if 0.5 <= t <= 0.95]
				ap_50_95 = sum(aps_range) / len(aps_range) if aps_range else 0.0
				per_class_map_50[cls_name] = round(ap_50, 2)
				per_class_map_50_95[cls_name] = round(ap_50_95, 2)

			run_summary["Per-Class mAP 50"] = per_class_map_50
			run_summary["Per-Class mAP 50:95"] = per_class_map_50_95

			self.summary[run_name] = run_summary.copy()

	def __evaluate_run(self, gt:Data, preds:Data, pos_threshold:float=None, background_threshold:float=None,
					   mode:str=AUTO, name:str=None, use_for_errors:bool=True) -> TIDERun:
		pos_thresh = self.pos_thresh if pos_threshold        is None else pos_threshold
		bg_thresh  = self.bg_thresh  if background_threshold is None else background_threshold
		mode       = self.mode       if mode                 is None else mode
		name       = preds.name      if name                 is None else name

		if mode == TIDE.AUTO:
			if gt.has_masks() and preds.has_masks():
				mode = TIDE.MASK 
			else:
				mode = TIDE.BOX

		run = TIDERun(gt, preds, pos_thresh, bg_thresh, mode, gt.max_dets, use_for_errors)

		if use_for_errors:
			self.runs[name] = run
		
		return run
	
	def average_out_confusion_matrices(self):
		cms = [run.confusion_matrix for run in self.runs.values() if hasattr(run, "confusion_matrix")]
		if len(cms) == 0:
			return None
		avg_cm = np.mean(cms, axis=0)
		self.avg_confusion_matrix = avg_cm
		return avg_cm

	def average_out_summary(self):
		all_summary = self.get_summary()
		averaged_summary = {}

		for summary in all_summary.values():
			for category_name, category_value in summary.items():
				if isinstance(category_value, float):
					if category_name not in averaged_summary:
						averaged_summary[category_name] = 0.0
					averaged_summary[category_name] += float(category_value)
				elif isinstance(category_value, dict):
					if category_name not in averaged_summary:
						averaged_summary[category_name] = {name: 0.0 for name in category_value.keys()}

					for name, value in category_value.items():
						averaged_summary[category_name][name] = float(value) + averaged_summary[category_name].get(name, 0)
				else:
					print("Type Mismatch! Can not average out the summaries.")
					break
		
		div_factor = len(all_summary.values())

		for category_name, category_value in averaged_summary.items():
			if isinstance(category_value, float):
				averaged_summary[category_name] = round(float(averaged_summary[category_name]) / div_factor, 2)
			elif isinstance(category_value, dict):
				for name, value in category_value.items():
					averaged_summary[category_name][name] = round(float(averaged_summary[category_name][name]) / div_factor, 2)
			else:
				print("Type Mismatch! Can not average out the summaries.")

		self.summary["Combined Average"] = averaged_summary.copy()

		self.average_out_confusion_matrices()


	def evaluate(self, gt:Data, preds:Data, mode:str=None, name:str=None):
		"""
		Evaluate a prediction set against a ground truth set.
		"""
		pos_threshold = self.pos_thresh
		if name          is None: name          = preds.name

		self.run_thresholds[name] = []

		for thresh in self.COCO_THRESHOLDS:
			
			run = self.__evaluate_run(gt, preds, pos_threshold=thresh, background_threshold=None,
				mode=mode, name=name, use_for_errors=(pos_threshold == thresh))
			
			self.run_thresholds[name].append(run)
		
		self.__calculate_summary()

	def evaluate_multiple_models_on_one_gt(self, gt:Data, preds_list:list, mode:str=None, names:list[str]=None):
		"""
		Evaluate multiple prediction sets against a ground truth set.
		preds_list: List of Data objects containing predictions
		names:      List of names for the prediction sets. If None, will use preds.name
		"""
		if names is None:
			names = [preds.name for preds in preds_list]
		elif len(names) != len(preds_list):
			raise ValueError("Length of names must match length of preds_list")
		
		for preds, name in zip(preds_list, names):
			self.evaluate(gt, preds, mode=mode, name=name)
		
		self.__calculate_summary()
	
	def evaluate_model_on_multiple_gt(self, gt_list:list[Data], preds_list:list[Data], mode:str=None, name:str=None):
		"""
		Evaluate prediction against multiple ground truth set.
		preds_list: List of Data objects containing predictions on multiple GTs
		gt_list: 	List of Data objects containing the GTs for the corresponding predictions
		mode: 	 	Evaluation mode (box or mask), Either str or list of str
		names:      List of names for the prediction sets. If None, will use preds.name
		"""
		if name is None:
			name = preds_list[0].name
		
		if isinstance(mode, str):
			mode = [mode] * len(preds_list)
		
		for gt_counter, (gt, preds) in enumerate(zip(gt_list, preds_list)):
			run_name = name + f"_GT_num_{gt_counter}"
			self.evaluate(gt, preds, name=run_name)
		
		self.__calculate_summary()
		self.average_out_summary()

	def get_summary(self) -> dict:
		"""
		Returns a summary of the mAP values and errors for all runs in this TIDE object.
		{
			run_name: {
				"mAP 50:95": float,
				'Threshold AP @': { AP Threshold: float },
				'main_errors': { error_name: float },
				'special_errors': { error_name: float }
			}
		}
		"""
		return self.summary.copy()

	def save_summary(self, out_path:str = "summary.json"):
		"""
		Saves a summary of the mAP values and errors for all runs in this TIDE object to out_path as a JSON file.
		"""
		f.save_json(self.get_summary(), out_path)
		# Zusätzlich: Speichere alle Confusion-Matrizen separat
		cm_out = {}
		for run_name, run in self.runs.items():
			if hasattr(run, "confusion_matrix"):
				cm_out[run_name] = {
					"labels": run.class_labels,
					"matrix": run.confusion_matrix.tolist()
				}

		if hasattr(self, "avg_confusion_matrix"):
			cm_out["Combined Average"] = {
				"labels": list(self.runs.values())[0].class_labels,
				"matrix": self.avg_confusion_matrix.tolist()
			}

		f.save_json(cm_out, out_path.replace(".json", "_confusion_matrices.json"))


	def print_summary(self):
		for run_name, run_summary in self.summary.items():
			print('-- {} --\n'.format(run_name))

			# mAP 50:95
			if "mAP 50:95" in run_summary:
				print('mAP 50:95: {:.2f}'.format(run_summary["mAP 50:95"]))

			# Threshold-APs
			if 'Threshold AP @' in run_summary and len(run_summary['Threshold AP @']) > 0:
				thresholds = list(run_summary['Threshold AP @'].keys())
				values = list(run_summary['Threshold AP @'].values())

				f.print_table([
					['Thresh'] + thresholds,
					['  AP  '] + values
				], title='Threshold AP @')

			# Main Errors
			if 'Main Errors' in run_summary and len(run_summary['Main Errors']) > 0:
				err_types = list(run_summary['Main Errors'].keys())
				err_vals = list(run_summary['Main Errors'].values())

				f.print_table([
					['Type'] + [e.short_name if hasattr(e, "short_name") else str(e) for e in err_types],
					[' dAP'] + err_vals
				], title='Main Errors')

			# Special Errors
			if 'Special Errors' in run_summary and len(run_summary['Special Errors']) > 0:
				spec_types = list(run_summary['Special Errors'].keys())
				spec_vals = list(run_summary['Special Errors'].values())

				f.print_table([
					['Type'] + [e.short_name if hasattr(e, "short_name") else str(e) for e in spec_types],
					[' dAP'] + spec_vals
				], title='Special Errors')

			if "Precision" in run_summary and "Recall" in run_summary:
				print(f"Precision: {float(run_summary["Precision"]["Average"]):.2f} | Recall: {float(run_summary["Recall"]["Average"]):.2f}")
			for size in ["Small", "Medium", "Large"]:
				ap_key, ar_key = f"AP ({size})", f"AR ({size})"
				if ap_key in run_summary["Precision"] and ar_key in run_summary["Recall"]:
					print(f"{ap_key}: {run_summary["Precision"][ap_key]} | {ar_key}: {run_summary["Recall"][ar_key]}")

			print()

	def print_confusion_matrices(self):
		for run_name, run in self.runs.items():
			if hasattr(run, "confusion_matrix"):
				print(f"\nConfusion Matrix ({run_name}):")
				print(run.class_labels)
				print(np.round(run.confusion_matrix, 2))


	def plot(self, out_dir:str=None):
		"""
		Plots a summary model for each run in the summary.
		Images will be outputted to out_dir, which will be created if it doesn't exist.
		"""
		if out_dir is None:
			out_dir = os.path.join(os.getcwd())
		
		f.plot(self.summary, out_dir)
	


	def get_main_errors(self):
		errors = {}

		for run_name, run in self.runs.items():
			if run_name in self.run_main_errors:
				errors[run_name] = self.run_main_errors[run_name]
			else:
				errors[run_name] = {
					error.short_name: value
						for error, value in run.fix_main_errors().items()
				}
		
		return errors.copy()

	def get_special_errors(self):
		errors = {}

		for run_name, run in self.runs.items():
			if run_name in self.run_special_errors:
				errors[run_name] = self.run_special_errors[run_name]
			else:
				errors[run_name] = {
					error.short_name: value
						for error, value in run.fix_special_errors().items()
				}
		
		return errors.copy()
	
	def get_all_errors(self):
		"""
		returns {
			'main'   : { run_name: { error_name: float } },
			'special': { run_name: { error_name: float } },
		}
		"""
		return {
			'main': self.get_main_errors(),
			'special': self.get_special_errors()
		}