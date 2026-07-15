import matplotlib.pyplot as plt
import numpy as np
import json
import os, sys
import seaborn as sns


def plot(d, out_dir: str):
	"""Backward-compatible entry point. Delegates to plotter.Plotter (the single,
	maintained plotting implementation) and writes the images directly into out_dir.

	Note: this writes into out_dir itself (it does NOT create an extra "Plots"
	subfolder), so pass the final directory you want the images in.
	"""
	from .plotter import Plotter
	os.makedirs(out_dir, exist_ok=True)
	Plotter._plot_single(d, out_dir)
	print("Plots saved!")

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
