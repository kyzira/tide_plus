from .data import Data
from .ap import ClassedAPDataObject
from .errors.main_errors import *
from .errors.qualifiers import Qualifier, AREA
from . import functions as f
from sklearn.metrics import confusion_matrix
from pycocotools import mask as mask_utils
from collections import defaultdict, OrderedDict
import numpy as np
from typing import Union

from tidecv.errors import qualifiers

class TIDEExample:
	""" Computes all the data needed to evaluate a set of predictions and gt for a single image. """
	def __init__(self, preds:list, gt:list, pos_thresh:float, mode:str, max_dets:int, run_errors:bool=True):
		self.preds          = preds
		self.gt             = [x for x in gt if not x['ignore']]
		self.ignore_regions = [x for x in gt if     x['ignore']]
		
		self.mode       = mode
		self.pos_thresh = pos_thresh
		self.max_dets   = max_dets
		self.run_errors = run_errors

		self._run()

	def _run(self):
		preds    = self.preds
		gt       = self.gt
		ignore   = self.ignore_regions
		det_type = 'bbox' if self.mode == 'bbox' else 'mask'
		max_dets = self.max_dets

		if len(preds) == 0:
			raise RuntimeError('Example has no predictions!')


		# Sort descending by score
		preds.sort(key=lambda pred: -pred['score'])
		preds = preds[:max_dets]
		self.preds = preds # Update internally so TIDERun can update itself if :max_dets takes effect
		detections = [x[det_type] for x in preds]

		
		# IoU is [len(detections), len(gt)]
		self.gt_iou = mask_utils.iou(
			detections,
			[x[det_type] for x in gt],
			[False] * len(gt))

		# Store whether a prediction / gt got used in their data list
		# Note: this is set to None if ignored, keep that in mind
		for idx, pred in enumerate(preds):
			pred['used'] = False
			pred['_idx'] = idx
			pred['iou']  = 0
		for idx, truth in enumerate(gt):
			truth['used']   = False
			truth['usable'] = False
			truth['_idx'] = idx

		pred_cls  = np.array([x['class'] for x in preds])
		gt_cls    = np.array([x['class'] for x in gt])

		if len(gt) > 0:
			# A[i,j] is true iff the prediction i is of the same class as gt j
			self.gt_cls_matching = (pred_cls[:, None] == gt_cls[None, :])
			self.gt_cls_iou = self.gt_iou * self.gt_cls_matching
			
			# This will be changed in the matching calculation, so make a copy
			iou_buffer = self.gt_cls_iou.copy()

			for pred_idx, pred_elem in enumerate(preds):
				# Find the max iou ground truth for this prediction
				gt_idx = np.argmax(iou_buffer[pred_idx, :])
				iou = iou_buffer[pred_idx, gt_idx]

				pred_elem['iou'] = np.max(self.gt_cls_iou[pred_idx, :])

				if iou >= self.pos_thresh:
					gt_elem = gt[gt_idx]

					pred_elem['used'] = True
					gt_elem['used']   = True
					pred_elem['matched_with'] = gt_elem['_id']
					gt_elem['matched_with']   = pred_elem['_id']

					# Make sure this gt can't be used again
					iou_buffer[:, gt_idx] = 0

		# Ignore regions annotations allow us to ignore predictions that fall within
		if len(ignore) > 0:
			# Because ignore regions have extra parameters, it's more efficient to use a for loop here
			for ignore_region in ignore:
				if ignore_region['mask'] is None and ignore_region['bbox'] is None:
					# The region should span the whole image
					ignore_iou = [1] * len(preds)
				else:
					if ignore_region[det_type] is None:
						# There is no det_type annotation for this specific region so skip it
						continue
					# Otherwise, compute the crowd IoU between the detections and this region
					ignore_iou = mask_utils.iou(detections, [ignore_region[det_type]], [True])

				for pred_idx, pred_elem in enumerate(preds):
					if not pred_elem['used'] and (ignore_iou[pred_idx] > self.pos_thresh) \
						and (ignore_region['class'] == pred_elem['class'] or ignore_region['class'] == -1):
						# Set the prediction to be ignored
						pred_elem['used'] = None

		if len(gt) == 0:
			return

		# Some matrices used just for error calculation
		if self.run_errors:
			self.gt_used = np.array([x['used'] == True for x in gt])[None, :]
			self.gt_unused = ~self.gt_used

			self.gt_unused_iou    = self.gt_unused     *  self.gt_iou
			self.gt_unused_cls    = self.gt_unused_iou *  self.gt_cls_matching
			self.gt_unused_noncls = self.gt_unused_iou * ~self.gt_cls_matching

			self.gt_noncls_iou    = self.gt_iou        * ~self.gt_cls_matching

			self.gt_used_iou      = self.gt_used       * self.gt_iou
			self.gt_used_cls      = self.gt_used_iou   * self.gt_cls_matching


class TIDERun:
	""" Holds the data for a single run of TIDE. """

	# Temporary variables stored in ground truth that we need to clear after a run
	_temp_vars = ['best_score', 'best_id', 'used', 'matched_with', '_idx', 'usable']
	

	def __init__(self, gt:Data, preds:Data, pos_thresh:float, bg_thresh:float, mode:str, max_dets:int, run_errors:bool=True):
		self.gt     = gt
		self.preds  = preds

		self._error_types = [ClassError, BoxError, OtherError, DuplicateError, BackgroundError, MissedError]
		self.errors     = []
		self.error_dict = {_type: [] for _type in self._error_types}
		self.ap_data = ClassedAPDataObject()
		self.qualifiers = {}

		# A list of false negatives per class
		self.false_negatives = {_id: [] for _id in self.gt.classes}

		self.pos_thresh = pos_thresh
		self.bg_thresh  = bg_thresh
		self.mode       = mode
		self.max_dets   = max_dets
		self.run_errors = run_errors

		self._run()

	def _compute_confusion_matrix(self):
		classes = sorted(self.gt.classes.keys())
		num_classes = len(classes)
		class_to_idx = {cls: i for i, cls in enumerate(classes)}
		bg_idx = num_classes  # Hintergrundklasse

		y_true_idx, y_pred_idx = [], []

		# Map image_id → GT- und Pred-Objekte
		gt_by_img = {img_id: self.gt.get(img_id) for img_id in self.gt.images}
		pred_by_img = {img_id: self.preds.get(img_id) for img_id in self.preds.images}

		for img_id in gt_by_img:
			gts = [g for g in gt_by_img[img_id] if not g["ignore"]]
			preds = pred_by_img.get(img_id, [])
			if len(gts) == 0 and len(preds) == 0:
				continue

			if len(gts) > 0 and len(preds) > 0:
				# Berechne IoU für alle Paare im Bild
				from pycocotools import mask as mask_utils
				det_type = "bbox" if self.mode == "bbox" else "mask"
				iou_matrix = mask_utils.iou(
					[p[det_type] for p in preds],
					[g[det_type] for g in gts],
					[False] * len(gts),
				)

				for p_idx, pred in enumerate(preds):
					best_gt_idx = int(np.argmax(iou_matrix[p_idx]))
					best_iou = iou_matrix[p_idx, best_gt_idx]

					if best_iou >= self.pos_thresh:
						gt_cls = gts[best_gt_idx]["class"]
						pred_cls = pred["class"]
						y_true_idx.append(class_to_idx[gt_cls])
						y_pred_idx.append(class_to_idx[pred_cls])
					else:
						# False positive
						y_true_idx.append(bg_idx)
						y_pred_idx.append(class_to_idx[pred["class"]])
			else:
				# Kein GT im Bild → alle Preds = False Positives
				for pred in preds:
					y_true_idx.append(bg_idx)
					y_pred_idx.append(class_to_idx[pred["class"]])
				# Kein Pred im Bild → alle GTs = False Negatives
				for gt in gts:
					y_true_idx.append(class_to_idx[gt["class"]])
					y_pred_idx.append(bg_idx)

		if not y_true_idx:
			return

		labels = list(range(num_classes)) + [bg_idx]
		cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels, normalize="true")
		self.confusion_matrix = cm
		self.class_labels = list(self.gt.classes.values()) + ["background"]



	def _run(self):
		""" And awaaay we go """

		for image in self.gt.images:
			x = self.preds.get(image)
			y = self.gt.get(image)

			# These classes are ignored for the whole image and not in the ground truth, so
			# we can safely just remove these detections from the predictions at the start.
			# However, since ignored detections are still used for error calculations, we have to keep them.
			if not self.run_errors:
				ignored_classes = self.gt._get_ignored_classes(image)
				x = [pred for pred in x if pred['class'] not in ignored_classes]

			self._eval_image(x, y)

		# Store a fixed version of all the errors for testing purposes
		for error in self.errors:
			error.original = f.nonepack(error.unfix())
			error.fixed    = f.nonepack(error.fix())
			error.disabled = False
		
		self.ap = self.ap_data.get_mAP()
		self._compute_confusion_matrix()

		self.precision_per_class = {}
		self.recall_per_class = {}
		self.ap_per_class = {}

		for cls_id, ap_obj in self.ap_data.objs.items():
			self.precision_per_class[self.gt.classes[cls_id]] = round(ap_obj.get_precision() * 100, 2)
			self.recall_per_class[self.gt.classes[cls_id]] = round(ap_obj.get_recall() * 100, 2)
			self.ap_per_class[self.gt.classes[cls_id]] = round(ap_obj.get_ap(), 2)

		self._clear()

	def _clear(self):
		"""Clears the ground truth and predictions so that they're ready for another run."""
		for gt in self.gt.annotations:
			for var in self._temp_vars:
				if var in gt:
					del gt[var]

		for pred in self.preds.annotations:
			for var in self._temp_vars + ['iou', 'info']:
				if var in pred:
					del pred[var]


	def _add_error(self, error):
		self.errors.append(error)
		self.error_dict[type(error)].append(error)

	def _eval_image(self, preds:list, gt:list):
		
		for truth in gt:
			if not truth['ignore']:
				self.ap_data.add_gt_positives(truth['class'], 1)

		if len(preds) == 0:
			# There are no predictions for this image so add all gt as missed
			for truth in gt:
				if not truth['ignore']:
					self.ap_data.push_false_negative(truth['class'], truth['_id'])

					if self.run_errors:
						self._add_error(MissedError(truth))
						self.false_negatives[truth['class']].append(truth)
			return

		ex = TIDEExample(preds, gt, self.pos_thresh, self.mode, self.max_dets, self.run_errors)
		preds = ex.preds # In case the number of predictions was restricted to the max

		for pred_idx, pred in enumerate(preds):

			pred['info'] = {'iou': pred['iou'], 'used': pred['used']}
			if pred['used']: pred['info']['matched_with'] = pred['matched_with']
			
			if pred['used'] is not None:
				self.ap_data.push(pred['class'], pred['_id'], pred['score'], pred['used'], pred['info'])
			
			# ----- ERROR DETECTION ------ #
			# This prediction is a negative (or ignored), let's find out why
			if self.run_errors and (pred['used'] == False or pred['used'] == None):
				# Test for BackgroundError
				if len(ex.gt) == 0: # Note this is ex.gt because it doesn't include ignore annotations
					# There is no ground truth for this image, so just mark everything as BackgroundError
					self._add_error(BackgroundError(pred))
					continue

				# Test for BoxError
				idx = ex.gt_cls_iou[pred_idx, :].argmax()
				if self.bg_thresh <= ex.gt_cls_iou[pred_idx, idx] <= self.pos_thresh:
					# This detection would have been positive if it had higher IoU with this GT
					self._add_error(BoxError(pred, ex.gt[idx], ex))
					continue

				# Test for ClassError
				idx = ex.gt_noncls_iou[pred_idx, :].argmax()
				if ex.gt_noncls_iou[pred_idx, idx] >= self.pos_thresh:
					# This detection would have been a positive if it was the correct class
					self._add_error(ClassError(pred, ex.gt[idx], ex))
					continue

				# Test for DuplicateError
				idx = ex.gt_used_cls[pred_idx, :].argmax()
				if ex.gt_used_cls[pred_idx, idx] >= self.pos_thresh:
					# The detection would have been marked positive but the GT was already in use
					suppressor = self.preds.annotations[ex.gt[idx]['matched_with']]
					self._add_error(DuplicateError(pred, suppressor))
					continue
					
				# Test for BackgroundError
				idx = ex.gt_iou[pred_idx, :].argmax()
				if ex.gt_iou[pred_idx, idx] <= self.bg_thresh:
					# This should have been marked as background
					self._add_error(BackgroundError(pred))
					continue

				# A base case to catch uncaught errors
				self._add_error(OtherError(pred))
		
		for truth in gt:
			# If the GT wasn't used in matching, meaning it's some kind of false negative
			if not truth['ignore'] and not truth['used']:
				self.ap_data.push_false_negative(truth['class'], truth['_id'])

				if self.run_errors:
					self.false_negatives[truth['class']].append(truth)
					
					# The GT was completely missed, no error can correct it
					# Note: 'usable' is set in error.py
					if not truth['usable']:
						self._add_error(MissedError(truth))
				


	def fix_errors(self, condition=lambda x: False, transform=None, false_neg_dict:dict=None,
				   ap_data:ClassedAPDataObject=None,
				   disable_errors:bool=False) -> ClassedAPDataObject:
		""" Returns a ClassedAPDataObject where all errors given the condition returns True are fixed. """
		if ap_data is None:
			ap_data = self.ap_data

		gt_pos = ap_data.get_gt_positives()
		new_ap_data = ClassedAPDataObject()

		# Potentially fix every error case
		for error in self.errors:
			if error.disabled:
				continue

			_id = error.get_id()
			_cls, data_point = error.original

			if condition(error):
				_cls, data_point = error.fixed
				
				if disable_errors:
					error.disabled = True
				
				# Specific for MissingError (or anything else that affects #GT)
				if isinstance(data_point, int):
					gt_pos[_cls] += data_point
					data_point = None
			
			if data_point is not None:
				if transform is not None:
					data_point = transform(*data_point)
				new_ap_data.push(_cls, _id, *data_point)
			
		# Add back all the correct ones
		for k in gt_pos.keys():
			for _id, (score, correct, info) in ap_data.objs[k].data_points.items():
				if correct:
					if transform is not None:
						score, correct, info = transform(score, correct, info)
					new_ap_data.push(k, _id, score, correct, info)

		# Add the correct amount of GT positives, and also subtract if necessary
		for k, v in gt_pos.items():
			# In case you want to fix all false negatives without affecting precision
			if false_neg_dict is not None and k in false_neg_dict:
				v -= len(false_neg_dict[k])
			new_ap_data.add_gt_positives(k, v)
			
		return new_ap_data

	def fix_main_errors(self, progressive:bool=False, error_types:list=None, qual:Qualifier=None) -> dict:
		ap_data = self.ap_data
		last_ap = self.ap

		if qual is None:
			qual = Qualifier('', None)

		if error_types is None:
			error_types = self._error_types

		errors = {}

		for error in error_types:
			_ap_data = self.fix_errors(qual._make_error_func(error),
				ap_data=ap_data, disable_errors=progressive)
			
			new_ap = _ap_data.get_mAP()
			# If an error is negative that means it's likely due to binning differences, so just
			# Ignore the negative by setting it to 0.
			errors[error] = max(new_ap - last_ap, 0)
			
			if progressive:
				last_ap = new_ap
				ap_data = _ap_data

		if progressive:
			for error in self.errors:
				error.disabled = False

		return errors
	
	def fix_special_errors(self, qual=None) -> dict:
		return {
			FalsePositiveError: self.fix_errors(transform=FalsePositiveError.fix).get_mAP()    - self.ap,
			FalseNegativeError: self.fix_errors(false_neg_dict=self.false_negatives).get_mAP() - self.ap}

	def count_errors(self, error_types:list=None, qual=None):
		counts = {}

		if error_types is None:
			error_types = self._error_types

		for error in error_types:
			if qual is None:
				counts[error] = len(self.error_dict[error])
			else:
				func = qualifiers.make_qualifier(error, qual)
				counts[error] = len([x for x in self.errors if func(x)])
		
		return counts


	def apply_qualifier(self, qualifier:Qualifier) -> ClassedAPDataObject:
		""" Applies a qualifier lambda to the AP object for this runs and stores the result in self.qualifiers. """

		pred_keep = defaultdict(lambda: set())
		gt_keep   = defaultdict(lambda: set())

		for pred in self.preds.annotations:
			if qualifier.test(pred):
				pred_keep[pred['class']].add(pred['_id'])
		
		for gt in self.gt.annotations:
			if not gt['ignore'] and qualifier.test(gt):
				gt_keep[gt['class']].add(gt['_id'])
		
		new_ap_data = self.ap_data.apply_qualifier(pred_keep, gt_keep)
		self.qualifiers[qualifier.name] = new_ap_data.get_mAP()
		return new_ap_data
		




