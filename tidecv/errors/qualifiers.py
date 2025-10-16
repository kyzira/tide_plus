# Defines qualifiers like "Extra small box"



def _area(x):
	if x is None:
		return 0
	# Wenn bbox vorhanden ist
	if x.get('bbox') is not None:
		b = x['bbox']
		return b[2] * b[3]
	# Wenn Maske vorhanden ist (z. B. bei Segmentation)
	if x.get('mask') is not None:
		import pycocotools.mask as mask_utils
		rle = x['mask']
		if rle is None:
			return 0
		try:
			return mask_utils.area(rle)
		except Exception:
			return 0
	# Standardwert
	return 0

def _ar(x):
	return x['bbox'][2] / x['bbox'][3]



class Qualifier():
	"""
	Creates a qualifier with the given name.

	test_func should be a callable object (e.g., lambda) that takes in as input an annotation
	object (either a ground truth or prediction) and returns whether or not that object qualifies (i.e., a bool).
	"""

	def __init__(self, name:str, test_func:object):
		self.test = test_func
		self.name = name
	
	# This is horrible, but I like it
	def _make_error_func(self, error_type):
		return (lambda err: isinstance(err, error_type) \
			and (self.test(err.gt) if hasattr(err, 'gt') else self.test(err.pred))) \
				if self.test is not None else (lambda err: isinstance(err, error_type))




AREA = [
	Qualifier('Small' , lambda x:           _area(x) <=  32 ** 2),
	Qualifier('Medium', lambda x: 32 ** 2 < _area(x) <=  96 ** 2),
	Qualifier('Large' , lambda x: 96 ** 2 < _area(x)            ),
]

ASPECT_RATIO = [
	Qualifier('Tall'  , lambda x:        _ar(x) <=  0.75),
	Qualifier('Square', lambda x: 0.75 < _ar(x) <=  1.33),
	Qualifier('Wide'  , lambda x: 1.33 < _ar(x)         ),
]


		
