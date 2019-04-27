#!/usr/bin/env python
import os.path
from glob import glob
import numpy as np

class DbController:
	#databases
	CHASE = 0
	STARE = 1
	HRF = 2
	def __init__(self):
		#zmienne do wczytania bazy
		self.sources = [{'name': "CHASE", 'org_ext': "jpg", 'res_ext1': "png", 'res_ext2': "png"},
						{'name': "STARE", 'org_ext': "ppm", 'res_ext1': "ppm"},
						{'name': "HRF",   'org_ext': "jpg", 'res_ext1': "tif", 'mask_ext': "tif"}]
		self.org_path = "original"
		self.res_path = "results"
		self.mask_path = "mask"


	def get_paths(self, db_nr, db_type, res_nr=1):
		db = self.sources[db_nr]

		if (db_type == 'ORG'):
			cat_path = self.org_path
			ext = 'org_ext'
		elif (db_type == 'MASK'):
			cat_path = self.mask_path
			ext = 'mask_ext'
		elif (db_type == 'RES'):
			cat_path = self.res_path + str(res_nr)
			ext = 'res_ext' + str(res_nr)
		else:
			raise ValueError('Wrong db_type.')

		here = os.path.dirname(os.path.realpath(__file__))
		micdrop = os.path.join(here, "resources", db['name'], cat_path) + "/*." + db[ext]
		return sorted(glob(micdrop))

	def get_dataset(self, db_nr, res_nr=1):
		org = self.get_paths(db_nr,'ORG')
		res = self.get_paths(db_nr,'RES')
		return np.array(list(zip(org,res)))


if __name__ == '__main__':
	test = DbController()
	print(test.get_dataset(DbController.STARE))
