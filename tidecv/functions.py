import matplotlib.pyplot as plt
import numpy as np
import json
import os, sys
import seaborn as sns


def plot(d, out_dir: str):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	plots_path = os.path.join(out_dir, "Plots")
	os.makedirs(plots_path, exist_ok=True)

	models = []
	map_values = {}
	threshold_values = {}
	main_errors = {}
	special_errors = {}

	for run_name, data in d.items():
		models.append(run_name)

		if "mAP 50:95" in data:
			map_values[run_name] = float(data["mAP 50:95"])

		if "Threshold AP @ " in data:
			data["Threshold AP @"] = data.pop("Threshold AP @ ")

		if "Threshold AP @" in data:
			threshold_values[run_name] = {
				str(k): float(v) for k, v in data["Threshold AP @"].items()
			}

		if "Main Errors" in data:
			main_errors[run_name] = {
				(str(e.short_name) if hasattr(e, "short_name") else str(e)): float(v)
				for e, v in data["Main Errors"].items()
			}

		if "Special Errors" in data:
			special_errors[run_name] = {
				(str(e.short_name) if hasattr(e, "short_name") else str(e)): float(v)
				for e, v in data["Special Errors"].items()
			}

	# --- 1. mAP50:95 Comparison ---
	if len(map_values) > 0:
		plt.figure()
		model_names = list(map_values.keys())
		values = [map_values[m] for m in model_names]
		plt.bar(model_names, values)
		plt.title("mAP 50:95 Comparison")
		plt.ylabel("mAP 50:95")
		plt.xlabel("Model")
		plt.tight_layout()
		plt.savefig(os.path.join(plots_path, "mAP_comparison.png"))
		plt.close()

	# --- 2. AP over Thresholds ---
	if len(threshold_values) > 0:
		plt.figure()
		for model, tvals in threshold_values.items():
			x = list(tvals.keys())
			y = list(tvals.values())
			plt.plot(x, y, marker='o', label=model)
		plt.title("AP over Thresholds (50–95)")
		plt.xlabel("Threshold")
		plt.ylabel("AP")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_path, "AP_thresholds_comparison.png"))
		plt.close()

	# --- 3. Main Errors (x=Model, Balken=Error Types) ---
	if len(main_errors) > 0:
		error_types = sorted({err for v in main_errors.values() for err in v.keys()})
		x = range(len(models))
		width = 0.8 / len(error_types)
		plt.figure()
		for i, err in enumerate(error_types):
			y = [main_errors[m].get(err, 0.0) for m in models]
			plt.bar([p + i*width for p in x], y, width, label=err)
		plt.xticks([p + width*(len(error_types)/2) for p in x], models, rotation=30)
		plt.title("Main Errors (dAP-Anteile)")
		plt.ylabel("dAP")
		plt.xlabel("Model")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_path, "Main_Errors_comparison.png"))
		plt.close()

	# --- 4. Special Errors (x=Model, Balken=Error Types) ---
	if len(special_errors) > 0:
		error_types = sorted({err for v in special_errors.values() for err in v.keys()})
		x = range(len(models))
		width = 0.8 / len(error_types)
		plt.figure()
		for i, err in enumerate(error_types):
			y = [special_errors[m].get(err, 0.0) for m in models]
			plt.bar([p + i*width for p in x], y, width, label=err)
		plt.xticks([p + width*(len(error_types)/2) for p in x], models, rotation=30)
		plt.title("Special Errors (dAP-Anteile)")
		plt.ylabel("dAP")
		plt.xlabel("Model")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_path, "Special_Errors_comparison.png"))
		plt.close()

	# --- 5. Precision (Average + Small/Medium/Large) ---
	precision_values = {}
	for run_name, data in d.items():
		if "Precision" in data:
			precision_values[run_name] = {
				k: float(v) for k, v in data["Precision"].items()
			}

	if len(precision_values) > 0:
		plt.figure()
		for model, vals in precision_values.items():
			keys = list(vals.keys())
			values = list(vals.values())
			plt.plot(keys, values, marker='o', label=model)
		plt.title("Precision Overview (Average / Size-based)")
		plt.xlabel("Category")
		plt.ylabel("Precision (%)")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_path, "Precision_comparison.png"))
		plt.close()

	# --- 6. Recall (Average + Small/Medium/Large) ---
	recall_values = {}
	for run_name, data in d.items():
		if "Recall" in data:
			recall_values[run_name] = {
				k: float(v) for k, v in data["Recall"].items()
			}

	if len(recall_values) > 0:
		plt.figure()
		for model, vals in recall_values.items():
			keys = list(vals.keys())
			values = list(vals.values())
			plt.plot(keys, values, marker='o', label=model)
		plt.title("Recall Overview (Average / Size-based)")
		plt.xlabel("Category")
		plt.ylabel("Recall (%)")
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_path, "Recall_comparison.png"))
		plt.close()

	# --- 7. Confusion Matrices ---
	confusion_data = {}
	for run_name, data in d.items():
		if "confusion_matrix" in data and "class_labels" in data:
			confusion_data[run_name] = {
				"cm": np.array(data["confusion_matrix"]),
				"labels": data["class_labels"]
			}

	if len(confusion_data) > 0:
		for model, cm_data in confusion_data.items():
			cm = cm_data["cm"]
			labels = cm_data["labels"]

			plt.figure(figsize=(8, 6))
			sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
						xticklabels=labels, yticklabels=labels, cbar=True)
			plt.title(f"Normalized Confusion Matrix ({model})")
			plt.xlabel("Predicted class")
			plt.ylabel("True class")
			plt.tight_layout()
			plt.savefig(os.path.join(plots_path, f"{model}_confusion_matrix.png"))
			plt.close()

	print(f"Plots saved!")

def print_table(rows:list, title:str=None):
	# Convert all elements to strings to avoid len() errors on floats or None
	rows = [[str(cell) for cell in row] for row in rows]

	# Get all rows to have the same number of columns
	max_cols = max([len(row) for row in rows])
	for row in rows:
		while len(row) < max_cols:
			row.append('')

	# Compute the text width of each column
	try:
		col_widths = [max([len(rows[i][col_idx]) for i in range(len(rows))]) for col_idx in range(len(rows[0]))]
	except Exception as e:
		print("Error computing column widths:", e)
		print("Rows were:")
		for row in rows:
			print(row)
		return

	divider = '--' + ('---'.join(['-' * w for w in col_widths])) + '-'
	thick_divider = divider.replace('-', '=')

	if title:
		left_pad = (len(divider) - len(title)) // 2
		print(('{:>%ds}' % (left_pad + len(title))).format(title))

	print(thick_divider)
	for row in rows:
		print('  ' + '   '.join([('{:>%ds}' % col_widths[col_idx]).format(row[col_idx]) for col_idx in range(len(row))]) + '  ')
		if row == rows[0]:
			print(divider)
	print(thick_divider)
	
def mean(arr:list):
	if len(arr) == 0:
		return 0
	return sum(arr) / len(arr)

def find_first(arr:np.array) -> int:
	""" Finds the index of the first instance of true in a vector or None if not found. """
	if len(arr) == 0:
		return None
	idx = arr.argmax()

	# Numpy argmax will return 0 if no True is found
	if idx == 0 and not arr[0]:
		return None
	
	return idx

def save_json(d:dict, out_path:str):
	with open(out_path, 'w') as f:
		json.dump(d, f, indent=4)

def isiterable(x):
	try:
		iter(x)
		return True
	except:
		return False

def recursive_sum(x):
	if isinstance(x, dict):
		return sum([recursive_sum(v) for v in x.values()])
	elif isiterable(x):
		return sum([recursive_sum(v) for v in x])
	else:
		return x

def apply_messy(x:list, func):
	return [([func(y) for y in e] if isiterable(e) else func(e)) for e in x]

def apply_messy2(x:list, y:list, func):
	return [[func(i, j) for i, j in zip(a, b)] if isiterable(a) else func(a, b) for a, b in zip(x, y)]

def multi_len(x):
	try:
		return len(x)
	except TypeError:
		return 1

def unzip(l):
	return map(list, zip(*l))


def points(bbox):
	bbox = [int(x) for x in bbox]
	return (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])

def nonepack(t):
	if t is None:
		return None, None
	else:
		return t


class HiddenPrints:
	""" From https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python """

	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout




def toRLE(mask:object, w:int, h:int):
	"""
	Borrowed from Pycocotools:
	Convert annotation which can be polygons, uncompressed RLE to RLE.
	:return: binary mask (numpy 2D array)
	"""
	import pycocotools.mask as maskUtils

	if type(mask) == list:
		# polygon -- a single object might consist of multiple parts
		# we merge all parts into one mask rle code
		rles = maskUtils.frPyObjects(mask, h, w)
		return maskUtils.merge(rles)
	elif type(mask['counts']) == list:
		# uncompressed RLE
		return maskUtils.frPyObjects(mask, h, w)
	else:
		return mask


def polyToBox(poly:list):
	""" Converts a polygon in COCO lists of lists format to a bounding box in [x, y, w, h]. """

	xmin = 1e10
	xmax = -1e10
	ymin = 1e10
	ymax = -1e10

	for poly_comp in poly:
		for i in range(len(poly_comp) // 2):
			x = poly_comp[2*i + 0]
			y = poly_comp[2*i + 1]

			xmin = min(x, xmin)
			xmax = max(x, xmax)
			ymin = min(y, ymin)
			ymax = max(y, ymax)
	
	return [xmin, ymin, (xmax - xmin), (ymax - ymin)]
