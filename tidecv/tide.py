from .data import Data
from .errors.main_errors import *
from . import functions as f
from collections import OrderedDict
from .quantify import TIDERun
import os
from .errors.qualifiers import Qualifier, AREA
from . import plotting as P

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

	def __init__(self, pos_threshold:float=0.5, background_threshold:float=0.1, mode:str=BOX):
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

		self.COCO_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

	def __calculate_summary(self):
		main_errors = self.get_main_errors()
		special_errors = self.get_special_errors()

		for run_name, run in self.runs.items():
			run_summary = {}
			if run_name in self.run_thresholds:
				thresh_runs = self.run_thresholds[run_name]
				aps = [trun.ap for trun in thresh_runs]
				
				run_summary["mAP 50:95"] = sum(aps)/len(aps)

				run_summary['Threshold AP @'] = {}
				for trun in thresh_runs:
					run_summary['Threshold AP @'][str(int(trun.pos_thresh*100))] = '{:6.2f}'.format(trun.ap)

				# --- Precision / Recall (gesamt) ---
			run_summary["Precision"] = {"Average": '{:6.2f}'.format(run.ap_data.get_precision() * 100)}
			run_summary["Recall"] = {"Average": '{:6.2f}'.format(run.ap_data.get_recall() * 100)}

			# --- AP / AR für Small, Medium, Large ---
			size_quals = {
				"Small": AREA[0],
				"Medium": AREA[1],
				"Large": AREA[2],
			}

			for size_name, qual in size_quals.items():
				ap_data = run.apply_qualifier(qual)
				run_summary["Precision"][f"AP ({size_name})"] = '{:6.2f}'.format(ap_data.get_mAP())
				run_summary["Recall"][f"AR ({size_name})"] = '{:6.2f}'.format(ap_data.get_recall() * 100)

			run_summary["Main Errors"] = {}
			for err in self._error_types:
				key = err.short_name if hasattr(err, "short_name") else str(err)
				if key in main_errors[run_name]:
					run_summary["Main Errors"][key] = '{:6.2f}'.format(main_errors[run_name][key])
				else:
					run_summary["Main Errors"][key] = '{:6.2f}'.format(0.0)

			run_summary["Special Errors"] = {}
			for err in self._special_error_types:
				key = err.short_name if hasattr(err, "short_name") else str(err)
				if key in special_errors[run_name]:
					run_summary["Special Errors"][key] = '{:6.2f}'.format(special_errors[run_name][key])
				else:
					run_summary["Special Errors"][key] = '{:6.2f}'.format(0.0)


			
			self.summary[run_name] = run_summary

	def __evaluate_run(self, gt:Data, preds:Data, pos_threshold:float=None, background_threshold:float=None,
					   mode:str=None, name:str=None, use_for_errors:bool=True) -> TIDERun:
		pos_thresh = self.pos_thresh if pos_threshold        is None else pos_threshold
		bg_thresh  = self.bg_thresh  if background_threshold is None else background_threshold
		mode       = self.mode       if mode                 is None else mode
		name       = preds.name      if name                 is None else name

		run = TIDERun(gt, preds, pos_thresh, bg_thresh, mode, gt.max_dets, use_for_errors)

		if use_for_errors:
			self.runs[name] = run
		
		return run

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

	def evaluate_multiple(self, gt:Data, preds_list:list, mode:str=None, names:list[str]=None):
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

				P.print_table([
					['Thresh'] + thresholds,
					['  AP  '] + values
				], title='Threshold AP @')

			# Main Errors
			if 'Main Errors' in run_summary and len(run_summary['Main Errors']) > 0:
				err_types = list(run_summary['Main Errors'].keys())
				err_vals = list(run_summary['Main Errors'].values())

				P.print_table([
					['Type'] + [e.short_name if hasattr(e, "short_name") else str(e) for e in err_types],
					[' dAP'] + err_vals
				], title='Main Errors')

			# Special Errors
			if 'Special Errors' in run_summary and len(run_summary['Special Errors']) > 0:
				spec_types = list(run_summary['Special Errors'].keys())
				spec_vals = list(run_summary['Special Errors'].values())

				P.print_table([
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
		
		return errors

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
		
		return errors
	
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