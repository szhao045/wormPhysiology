# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:34:07 2015

@author: Willie
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import scipy.ndimage
import warnings
import pathlib
import concurrent.futures
import multiprocessing
import pickle
import gc
import shutil
import matplotlib.pyplot as plt

import freeimage
from zplib.image import mask as zplib_image_mask

import basicOperations.folderStuff as folderStuff
import basicOperations.imageOperations as imageOperations
import wormFinding.backgroundSubtraction as backgroundSubtraction
import wormFinding.textureClassification as textureClassification
import wormFinding.edgeMorphology as edgeMorphology
import measurePhysiology.extractFeatures as extractFeatures

warnings.filterwarnings('ignore')

gui_called = False

class BaseMeasurer():
	def __init__(self, experiment_directory, working_directory, validated_directory, print_prefix = ''):
		print(print_prefix + 'Initializing base_measurer...')
		self.print_prefix = print_prefix
		self.t0 = time.time()
		self.experiment_directory = experiment_directory
		self.working_directory = working_directory
		self.worm_dirs = sorted([experiment_directory + os.path.sep + an_item for an_item in os.listdir(experiment_directory) if os.path.isdir(experiment_directory + os.path.sep + an_item) and an_item[0] in '0123456789'])
		with open(experiment_directory + os.path.sep + 'experiment_metadata.json', 'r') as read_file:
			self.metadata = json.loads(read_file.read())
		self.timepoints = self.metadata['timepoints']
		self.validated_directory = validated_directory
		self.validated_points = []
		for a_subdir in [an_object for an_object in os.listdir(validated_directory) if os.path.isdir(validated_directory + os.path.sep + an_object)]:
			for a_file in os.listdir(validated_directory + os.path.sep + a_subdir):
				if a_file.split(' ')[-1] == 'hmask.png':
					self.validated_points.append(validated_directory + os.path.sep + a_subdir + os.path.sep + a_file.split(' ')[0])
		self.bf_reference_intensity = 11701.7207031
		self.calibration_directory = experiment_directory + os.path.sep + 'calibrations'		
		print(print_prefix + '\tOrganized some information, took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		
		# Make super vignette if needed.
		self.t0 = time.time()		
		self.super_vignette = self.make_super_vignette()
		print(print_prefix + '\tMade a mega-vignette mask, took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)	
		
		# Grid search (if necessary) and train SVMs that we'll use.
		self.t0 = time.time()
		if os.path.exists(self.validated_directory + os.path.sep + 'hyperparameters.json'):
			with open(self.validated_directory + os.path.sep + 'hyperparameters.json', 'r') as read_file:
				self.known_settings = json.loads(read_file.read())
		else:
			self.known_settings = {}
			
		self.main_classifier = 'bf;Local_Worm;10000;5;Intensity;8-Star'
		self.grid_search_texture_classifier([self.main_classifier])
		self.t0 = time.time()
		samples_per_image = 200
		self.age_texture_codebook = extractFeatures.train_texture(validated_directory, regression_variable = 'Age')		
		(self.age_texture_regressor, self.known_settings) = extractFeatures.train_texture_SVM(validated_directory, self.age_texture_codebook, samples_per_image = samples_per_image, regression_variable = 'Age', hyperparameters = self.known_settings, validated_directory = self.validated_directory)
		self.ghost_texture_codebook = extractFeatures.train_texture(validated_directory, regression_variable = 'Ghost_Age')		
		(self.life_texture_regressor, self.known_settings) = extractFeatures.train_texture_SVM(validated_directory, self.ghost_texture_codebook, samples_per_image = samples_per_image, regression_variable = 'Ghost_Age', hyperparameters = self.known_settings, validated_directory = self.validated_directory)
		self.egg_texture_codebook = extractFeatures.train_texture(validated_directory, regression_variable = 'Egg_Age')		
		(self.egg_texture_regressor, self.known_settings) = extractFeatures.train_texture_SVM(validated_directory, self.egg_texture_codebook, samples_per_image = samples_per_image, regression_variable = 'Egg_Age', hyperparameters = self.known_settings, validated_directory = self.validated_directory)
		print(print_prefix + '\tTrained texture classification codebook and SVMs, took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)	
		self.trained_classifiers = {}
		self.train_all_SVMs()
		self.worm_parameters = textureClassification.string_to_from_dictionary(self.main_classifier)
		self.worm_classifier = self.trained_classifiers[self.main_classifier]		
		print(print_prefix + '\tTrained all the SVMs that we\'ll use!', flush = True)	
		
		# Make great lawns.
		for worm_subdir in self.worm_dirs:
			if not os.path.isfile(worm_subdir + os.path.sep + 'great_lawn.png'):
				self.t0 = time.time()
				edgeMorphology.make_mega_lawn(worm_subdir, self.super_vignette.copy())
				print('Made great lawn for ' + str(worm_subdir) + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		print(print_prefix + '\tEnsured all the great lawns that we\'ll use!', flush = True)

	def make_super_vignette(self):
		'''
		Combines all the vignette masks in calibration_directory to create a more comprehensive single vignette mask.
		'''
		pickle_file = self.experiment_directory + os.path.sep + 'super_vignette.pickle'
		if os.path.isfile(pickle_file):
			with open(pickle_file, 'rb') as my_file:		
				my_vignette = pickle.load(my_file)
				return my_vignette
				
		my_masks = [self.calibration_directory + os.path.sep + a_file for a_file in os.listdir(self.calibration_directory) if 'vignette_mask.png' in a_file]
		mask_shape = freeimage.read(my_masks[0]).shape
		my_vignette = np.ones(mask_shape).astype('bool')
		for a_mask in my_masks:
			my_vignette[np.invert(freeimage.read(a_mask).astype('bool'))] = False
		my_vignette[0, :] = False
		my_vignette[-1, :] = False
		my_vignette[:, 0] = False
		my_vignette[:, -1] = False
		my_distances = scipy.ndimage.morphology.distance_transform_edt(my_vignette)
		my_vignette[my_distances < 50] = False

		with open(pickle_file, 'wb') as my_file:		
			pickle.dump(my_vignette, my_file)	
		return my_vignette

	def grid_search_texture_classifier(self, parameters_list):
		'''
		Optimize hyperparameters to train an SVM for texture classification for better masking.
		'''
		for parameter_string in parameters_list:
			SVM_external_parameters = textureClassification.string_to_from_dictionary(parameter_string)
			if parameter_string not in self.known_settings.keys():
				# Find the best hyperparameters.
				texture_svr_tuple = textureClassification.train_texture_SVM(self.validated_points, SVM_external_parameters, self.super_vignette)
				best_params = textureClassification.do_SVM_grid_search(texture_svr_tuple, SVM_external_parameters)	

				# Get a sense of how good the classifier is.
				(texture_SVM, feature_array, score_array) = textureClassification.train_texture_SVM(self.validated_points, SVM_external_parameters, self.super_vignette, hyperparameters = best_params)
				try_answers = []
				for i in range(len(score_array)):
					a_patch = feature_array[i]		
					try_answers.append(texture_SVM.predict(a_patch)) 		
				try_answers = np.array(try_answers)[:, 0]
				my_percentage = 1 - np.sum(np.abs(score_array - try_answers))/try_answers.shape[0]
				best_params['accuracy'] = my_percentage
				self.known_settings[parameter_string] = best_params
				with open(self.validated_directory + os.path.sep + 'hyperparameters.json', 'w') as write_file:
					 write_file.write(json.dumps(self.known_settings, sort_keys=True, indent=4))
		return

	def train_all_SVMs(self):
		'''
		Train all the SVMs that we're gonna need. They should all have their hyperparameters already figured out!
		'''
		parameters_list = [
			self.main_classifier
		]
		
		for i in range(0, len(parameters_list)):
			self.t0 = time.time()
			parameter_string = parameters_list[i]
			pickle_file = self.validated_directory + os.path.sep + parameter_string + 'SVC.pickle'
			if os.path.isfile(pickle_file):
				with open(pickle_file, 'rb') as my_file:		
					texture_SVC = pickle.load(my_file)
			else:
				SVM_external_parameters = textureClassification.string_to_from_dictionary(parameter_string)
				best_params = self.known_settings[parameter_string]
				(texture_SVC, feature_array, score_array) = textureClassification.train_texture_SVM(self.validated_points, SVM_external_parameters, self.super_vignette, hyperparameters = best_params)
				with open(pickle_file, 'wb') as my_file:		
					pickle.dump(texture_SVC, my_file)	

			self.trained_classifiers[parameter_string] = texture_SVC
			print(self.print_prefix + '\tTrained SVM' + ' for ' + str(i+1) + '/' + str(len(parameters_list)) + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		return
	
	def classify_image(self, bf_image, rough_mask, parameter_string, threaded = True, extra_distance = None):
		'''
		Use a single SVM to refine a known worm mask.		
		'''
		best_params = self.known_settings[parameter_string]
		SVM_external_parameters = textureClassification.string_to_from_dictionary(parameter_string)
		if parameter_string not in self.trained_classifiers.keys():
			(texture_SVM, feature_array, score_array) = textureClassification.train_texture_SVM(self.validated_points, SVM_external_parameters, self.super_vignette, hyperparameters = best_params)
			self.trained_classifiers[parameter_string] = texture_SVM
		texture_SVM = self.trained_classifiers[parameter_string]
		refined_mask = textureClassification.classify_worm_nonworm(bf_image, rough_mask, texture_SVM, self.super_vignette, SVM_external_parameters, threaded, extra_distance)
		if np.count_nonzero(refined_mask) > 0:
			refined_mask[np.invert(zplib_image_mask.get_largest_object(refined_mask))] = False	
			refined_mask = zplib_image_mask.fill_small_area_holes(refined_mask, 50).astype('bool')
		return refined_mask

class WormMeasurer(BaseMeasurer):
	def __init__(self, worm_subdirectory, working_directory, validated_directory):
		print('Measuring the worm recorded in ' + worm_subdirectory + '...', flush = True)		
		super().__init__(str(pathlib.Path(worm_subdirectory).parent), working_directory, validated_directory, print_prefix = '\t')
		self.worm_times = [a_time for a_time in self.timepoints if os.path.isfile(worm_subdirectory + os.path.sep + a_time + ' ' + 'bf.png')]
		self.temporal_radius = 10
		self.worm_subdirectory = worm_subdirectory	
		self.worm_name = self.worm_subdirectory.replace(self.experiment_directory + os.path.sep, '')
		experiment_name = self.worm_subdirectory.split(os.path.sep)[-2].replace(' Run ', ' ')
		self.write_directory = self.working_directory + os.path.sep + experiment_name + ' ' + self.worm_name
		folderStuff.ensure_folder(self.write_directory)
		self.full_worm_name = experiment_name + ' ' + self.worm_name
		with open(worm_subdirectory + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
			self.position_metadata = json.loads(read_file.read())
		shutil.copyfile(worm_subdirectory + os.path.sep + 'position_metadata_extended.json', self.write_directory + os.path.sep + 'position_metadata_extended.json')
		self.bacterial_lawn = freeimage.read(worm_subdirectory + os.path.sep + 'great_lawn.png').astype('bool')
		self.worm_frame = pd.DataFrame(index = self.worm_times, columns = [
			'intensity_50', 'intensity_60', 'intensity_70', 'intensity_80', 'intensity_90', 'intensity_95', 'intensity_100', 'integrated_50', 'integrated_60', 'integrated_70', 'integrated_80', 'integrated_90', 'integrated_95', 'integrated_0', 
			'age_texture', 'egg_texture', 'life_texture', 'intensity_texture',
			'unstimulated_rate', 'stimulated_rate_a', 'stimulated_rate_b', 'bulk_movement', 'fine_movement',
			'total_size', 'aspect_ratio',
			'visible_area', 'visible_eggs', 'average_egg_size', 'single_eggs',
			'great_lawn_area', 'great_lawn_max_diameter',
			'age', 'egg_age', 'ghost_age'
		])	

	def read_corrected_bf(self, time_point, movement_key = ''):
		'''
		Read in an image at time_point and properly correct it for flatfield and metering.
		'''
		raw_image = freeimage.read(self.worm_subdirectory + os.path.sep + time_point + ' ' + 'bf' + movement_key + '.png')		
		flatfield_image = freeimage.read(self.calibration_directory + os.path.sep + time_point + ' ' + 'bf_flatfield.tiff')
		time_reference = self.metadata['brightfield metering'][time_point]['ref_intensity']
		raw_image[np.invert(self.super_vignette)] = 0
		corrected_image = raw_image*flatfield_image	
		corrected_image = corrected_image / time_reference * self.bf_reference_intensity
		corrected_image = corrected_image.astype('uint16')
		return corrected_image	
	
	def read_corrected_fluorescence(self, time_point, hot_threshold = 500):
		'''
		Correct fluorescence images for flatfield, and re-normalize to make the images more nicely viewable.	
		'''
		# Read in image and apply vignetting.
		image_path = self.worm_subdirectory + os.path.sep + time_point + ' ' + 'green_yellow_excitation_autofluorescence.png'
		raw_image = freeimage.read(image_path)
		raw_image[np.invert(self.super_vignette)] = 0

		# Correct for flatfield.
		flatfield_path = self.calibration_directory + os.path.sep + time_point + ' ' + 'fl_flatfield.tiff'
		calibration_image = freeimage.read(flatfield_path)
		corrected_image = raw_image*calibration_image	

		# Correct for hot pixels.
		median_image = scipy.ndimage.filters.median_filter(corrected_image, size = 3)
		difference_image = np.abs(corrected_image.astype('float64') - median_image.astype('float64')).astype('uint16')
		hot_pixels = difference_image > hot_threshold
		median_image_hot_pixels = median_image[hot_pixels]
		corrected_image[hot_pixels] = median_image_hot_pixels

		# Return the actual image.
		return corrected_image
	
	def write_results(self, my_time, focal_frame, background_frame, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored, write_health = True):
		'''
		Write out my results to disk as I go. This is how things should go!
		'''
		# Prepare directories and a base name to write to for the time point.
		os.makedirs(self.working_directory, exist_ok = True)
		os.makedirs(self.write_directory, exist_ok = True)
		base_name = self.write_directory + os.path.sep + my_time + ' '

		# Write out my images.
		freeimage.write(focal_frame.astype('uint16'), base_name + 'bf.png')
		freeimage.write(imageOperations.renormalize_image(focal_mask.astype('uint8')), base_name + 'mask.png')
		freeimage.write(imageOperations.renormalize_image(focal_fluorescence.astype('uint16')), base_name + 'fluorescence.png')
		freeimage.write(background_frame, base_name + 'background.png')
		freeimage.write(worm_eggs_colored, base_name + 'color_worm_eggs.png')
		freeimage.write(fluorescence_colored, base_name + 'color_fluorescence.png')
		freeimage.write(movement_colored, base_name + 'color_movement.png')
		freeimage.write(bulk_move_colored, base_name + 'color_bulk_movement.png')
		
		# Write out the measured data.
		if write_health:
			final_directory = self.experiment_directory + os.path.sep + 'measured_health'
			os.makedirs(final_directory, exist_ok = True)
			self.worm_frame.to_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t')
		return

	def add_metainformation(self):
		'''
		Add information about worm age, birth, death, and maturity to our final measured health dataframe.
		'''
		# Ensure that my output directory exists.
		final_directory = self.experiment_directory + os.path.sep + 'measured_health'
		os.makedirs(final_directory, exist_ok = True)
		
		# Add in the important elements!
		self.worm_frame = pd.read_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t', index_col = 0)
		great_lawn_data = extractFeatures.measure_great_lawn(self.bacterial_lawn)
		for timepoint in list(self.worm_frame.index):
			i = self.timepoints.index(timepoint)
			self.worm_frame.loc[self.timepoints[i], 'age'] = self.position_metadata[i]['age']
			self.worm_frame.loc[self.timepoints[i], 'egg_age'] = self.position_metadata[i]['egg_age']
			self.worm_frame.loc[self.timepoints[i], 'ghost_age'] = self.position_metadata[i]['ghost_age']
			self.worm_frame.loc[self.timepoints[i], ['great_lawn_area', 'great_lawn_max_diameter']] = great_lawn_data
		self.worm_frame.to_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t')
		return
	
	def mask_decision(self, my_time, focal_frame, background_mask, verbose_write = False, print_suffix = ''):
		'''
		Generate a final mask based on metadata information and my three masks.
		'''		
		def write_mask(a_mask, a_name):
			'''
			Write out a mask with a_name.
			'''
			base_name = self.write_directory + os.path.sep + my_time + ' '
			write_mask = a_mask.copy()
			write_mask = write_mask.astype('uint8')
			write_mask[write_mask > 0] = -1
			freeimage.write(write_mask, base_name + a_name + '.png')
			return

		def get_size_and_center(mask_dictionary, reference_size = self.last_size, reference_mask = 'background'):
			'''
			Measure some variables that we use to locate the best mask from our choices.
			'''
			distance_dictionary = {}
			size_dictionary = {}
			center_dictionary = {}
			reference_pixels = np.array(np.where(mask_dictionary[reference_mask] > 0))			
			reference_center = np.mean(reference_pixels, axis = 1)
			for a_mask in mask_dictionary.keys():
				try:
					worm_pixels = np.array(np.where(mask_dictionary[a_mask] > 0))
					my_center = np.mean(worm_pixels, axis = 1)
					center_dictionary[a_mask] = my_center
					my_distance = np.linalg.norm(reference_center - my_center)
					distance_dictionary[a_mask] = my_distance
					size_dictionary[a_mask] = worm_pixels.shape[1]
				except:
					if np.count_nonzero(mask_dictionary[a_mask]) == 0:
						center_dictionary[a_mask] = (0, 0)
						distance_dictionary[a_mask] = 999
						size_dictionary[a_mask] = 0
					else:
						raise ValueError('Can\t get size and center of ' + a_mask + '!')	
			return (distance_dictionary, size_dictionary, center_dictionary)

		# Initialize some variables that we'll need.		
		time_index = self.timepoints.index(my_time)	
		mask_dictionary = {'background': background_mask}
		
		# Compute edge masks.
		self.t0 = time.time()
		(eggs_mask, mask_dictionary['edge'], my_edges) = edgeMorphology.find_eggs_worm(focal_frame, self.super_vignette, self.bacterial_lawn, worm_sigma = 2.0)
		print('\tComputed edge mask for ' + my_time + print_suffix + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Compute SVM masks.	
		self.t0 = time.time()
		mask_dictionary['maximum'] = mask_dictionary['edge'] | mask_dictionary['background']
		mask_dictionary['SVM'] = self.classify_image(focal_frame, mask_dictionary['maximum'], self.main_classifier, threaded = False, extra_distance = 0).astype('bool')
		mask_dictionary['edge_svm'] = mask_dictionary['edge'] & mask_dictionary['SVM']
		mask_dictionary['background_svm'] = mask_dictionary['background'] & mask_dictionary['SVM']
		mask_dictionary['minimum'] = mask_dictionary['background_svm'] & mask_dictionary['edge_svm']
		print('\tComputed SVM masks for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Check if the worm is mature or not and see how good some of my masks are.
		self.t0 = time.time()	
		if self.position_metadata[time_index]['egg_age'] >= 0:
			worm_mature = True
		else:
			worm_mature = False
		(distance_dictionary, size_dictionary, center_dictionary) = get_size_and_center(mask_dictionary)	
		
		# Make backup edges if necessary.
		if distance_dictionary['edge'] > 75:		
			backup_edges = edgeMorphology.backup_worm_find(my_edges, center_dictionary['background'])
			if backup_edges != None:
				mask_dictionary['backupEdges'] = backup_edges
				(distance_dictionary, size_dictionary, center_dictionary) = get_size_and_center(mask_dictionary)		

		# Make the actual decision and return our focal mask. It defaults to the background subtraction mask to start out.
		i = 0
		candidate_order = [
			'edge_svm',
			'edge',
			'backupEdges',
			'background_svm',
			'background'
		]
		focal_mask = mask_dictionary['background']
		mask_selected = False
		while not mask_selected:
			a_candidate = candidate_order[i]
			if a_candidate in mask_dictionary.keys():
				if (distance_dictionary[a_candidate] < 75 or not worm_mature) and size_dictionary[a_candidate] >= 0.50*self.last_size and size_dictionary[a_candidate] < 100000:
					mask_selected = True
					focal_mask = mask_dictionary[a_candidate]
			i += 1
			if i >= len(candidate_order):
				mask_selected = True				
		print('\tMade final decision for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Optionally write out everything.
		if verbose_write or not verbose_write:
			write_mask(my_edges, 'my_edges')
			for a_mask in mask_dictionary.keys():
				write_mask(mask_dictionary[a_mask], 'mask_de_' + a_mask)
		return (focal_mask, eggs_mask)
		
	def make_measurements(self, my_time, last_mask, focal_mask, movement_masks, temporal_radius, i, last_frame, focal_frame, eggs_mask):
		'''
		Given the positions of the worm and eggs and raw data files, make the actual measurements that I want.
		'''
		# Measure movement, one way or another.
		self.t0 = time.time()			
		if my_time == self.timepoints[temporal_radius]:
				(self.worm_frame.loc[my_time], movement_colored, bulk_move_colored) = extractFeatures.measure_movementCOM(movement_masks, [focal_mask, focal_mask], [focal_frame, focal_frame], self.position_metadata[i], self.worm_frame.loc[my_time]) 
		else:
			(self.worm_frame.loc[my_time], movement_colored, bulk_move_colored) = extractFeatures.measure_movementCOM(movement_masks, [last_mask, focal_mask], [last_frame, focal_frame], self.position_metadata[i], self.worm_frame.loc[my_time]) 
				
		# Measure size and set up last_mask.
		last_mask = focal_mask.copy()		
		last_frame = focal_frame.copy()
		self.worm_frame.loc[my_time] = extractFeatures.measure_size(focal_mask, self.worm_frame.loc[my_time])
		
		# Only do autofluorescence measurements if fluorescence was on for this time point.
		if os.path.isfile(self.worm_subdirectory + os.path.sep + my_time + ' ' + 'green_yellow_excitation_autofluorescence.png'):
			focal_fluorescence = self.read_corrected_fluorescence(my_time)
			(self.worm_frame.loc[my_time], fluorescence_colored) = extractFeatures.measure_autofluorescence(focal_fluorescence, focal_mask, self.worm_frame.loc[my_time])
		else:
			focal_fluorescence = np.zeros(focal_frame.shape).astype('uint8')
			fluorescence_colored = focal_fluorescence
		print('\tMeasured movement, size, and fluorescence for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Measure the various textures.
		self.t0 = time.time()			
		self.worm_frame.loc[my_time] = extractFeatures.measure_texture(focal_frame, focal_mask, [self.age_texture_codebook, self.egg_texture_codebook, self.ghost_texture_codebook], [self.age_texture_regressor, self.egg_texture_regressor, self.life_texture_regressor], self.worm_frame.loc[my_time])
		print('\tMeasured texture for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Measure the eggs.
		self.t0 = time.time()			
		(self.worm_frame.loc[my_time], worm_eggs_colored) = extractFeatures.measure_eggs(eggs_mask, focal_mask, self.worm_frame.loc[my_time])
		print('\tMeasured eggs for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		return (last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored)
		
	def measure_one_time(self, my_time):
		'''
		Measure a single time point for this worm. It is used on its own.
		'''
		print('Measuring ' + my_time + ', ' + self.full_worm_name + '.', flush = True)
		# Find the worm!!! This part is different between measure_one_time and measure_a_time.			
		os.makedirs(self.write_directory + os.path.sep + 'test_dir', exist_ok = True)
		self.write_directory = self.write_directory + os.path.sep + 'test_dir'

		self.last_size = 1
		self.t1 = time.time()
		self.t0 = time.time()
		i = self.timepoints.index(my_time)
		last_index =  i - 1
		last_time = self.timepoints[last_index]
		last_frame = self.read_corrected_bf(last_time)
		last_mask = freeimage.read(self.working_directory + os.path.sep + self.full_worm_name + os.path.sep + last_time + ' ' + 'mask.png').astype('bool')
		focal_frame = self.read_corrected_bf(my_time)
		movement_frames = [self.read_corrected_bf(my_time, movement_key = movement_key) for movement_key in ['00', '01', '10', '11', '12']]
		background_model = freeimage.read(self.working_directory + os.path.sep + self.full_worm_name + os.path.sep + last_time + ' ' + 'background.png')

		# Write out my background subtraction masks.
		(background_mask, movement_masks) = backgroundSubtraction.subtract_one_time(background_model, focal_frame, movement_frames, self.bacterial_lawn)
		print('\tGot masks from background subtraction for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Find the worm!!! This part is the same for both measure_one_time and measure_a_time.
		(focal_mask, eggs_mask) = self.mask_decision(my_time, focal_frame, background_mask, verbose_write = True)	
		print('\tGot masks from background subtraction, edge detection, and texture classification for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t1)[:5] + ' seconds.', flush = True)
		
		# Make all my measurements.
		temporal_radius = 10
		(last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored) = self.make_measurements(my_time, last_mask, focal_mask, movement_masks, temporal_radius, i, last_frame, focal_frame, eggs_mask)		
		
		# Write out my results.
		self.t0 = time.time()	
		self.write_results(my_time, focal_frame, background_model, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored, write_health = False)
		print('\tWrote results for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)		
		return
		
	def measure_a_time(self, current_context, my_time, last_frame, last_mask, i, temporal_radius):
		'''
		Measure a time point for this worm. It is called by self.measure_a_worm().
		'''
		print(str(i+1) + '/' + str(self.max_legitimate_times) + ': Measuring ' + my_time + ', ' + self.full_worm_name + '.', flush = True)

		# Find the worm!!! This part is different between measure_one_time and measure_a_time.			
		self.t1 = time.time()
		self.t0 = time.time()
		focal_frame = self.read_corrected_bf(my_time)
		movement_frames = [self.read_corrected_bf(my_time, movement_key = movement_key) for movement_key in ['00', '01', '10', '11', '12']]
		(current_context, background_model, background_mask, movement_masks) = backgroundSubtraction.background_frame(current_context, focal_frame, movement_frames, self.bacterial_lawn, i) 		
		print('\tGot masks from background subtraction for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Find the worm!!! This part is the same for both measure_one_time and measure_a_time.
		(focal_mask, eggs_mask) = self.mask_decision(my_time, focal_frame, background_mask)
		print('\tGot final masks for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t1)[:5] + ' seconds.', flush = True)

		# Make all my measurements.
		(last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored) = self.make_measurements(my_time, last_mask, focal_mask, movement_masks, temporal_radius, i, last_frame, focal_frame, eggs_mask)		
		self.last_size = min(self.worm_frame.loc[my_time, 'total_size'], 100000)
		
		# Write out my results.
		self.t0 = time.time()	
		self.write_results(my_time, focal_frame, background_model, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored)
		print('\tWrote results for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)		
		return (current_context, last_frame, last_mask)

	def measure_a_worm_try(self, temporal_radius = 10):
		'''
		Try to fully measure a worm! If it fails, throw a fit so you get noticed!!! 
		'''	
		try:
			self.measure_a_worm()
		except:
			while True:
				print(self.full_worm_name + ' HAS FAILED!!!!!!!!!!!! PANIC!!!!!!!!!!!!!!!!!!!!', flush = True)
		return
	
	def measure_a_worm(self, temporal_radius = 10):
		'''
		Fully measure a worm!
		'''	
		# Run background subtraction forward.
		self.last_size = 1
		death_frame = [i for i in range(0, len(self.position_metadata)) if abs(self.position_metadata[i]['ghost_age']) < 0.1][0] + 1
		self.max_legitimate_times = min(len(list(self.worm_frame.index)), len(self.position_metadata), death_frame)
		original_context = [self.read_corrected_bf(a_time) for a_time in self.worm_times[:temporal_radius]]
		context_frames = list(original_context)
		print('Starting to measure ' + self.full_worm_name + '.', flush = True)
		last_mask = None
		last_frame = None
		for i in range(temporal_radius, self.max_legitimate_times):
			(context_frames, last_frame, last_mask) = self.measure_a_time(context_frames, self.worm_times[i], last_frame, last_mask, i, temporal_radius)			
			if i == temporal_radius:
				radius_mask = last_mask.copy()
				radius_frame = last_frame.copy()

		# Run background subtraction backward to get the last few files.				
		original_context = [self.read_corrected_bf(a_time) for a_time in reversed(self.worm_times[temporal_radius:temporal_radius*2])]
		context_frames = list(original_context)
		last_frame = radius_frame
		last_mask = radius_mask
		for i in reversed(range(0, temporal_radius)):
			(context_frames, last_frame, last_mask) = self.measure_a_time(context_frames, self.worm_times[i], last_frame, last_mask, i, temporal_radius)
		
		self.add_metainformation()
		return

class HumanWormMeasurer(BaseMeasurer):
	def __init__(self, worm_subdirectory, human_subdirectory, working_directory, validated_directory):
		print('Measuring the worm recorded in ' + worm_subdirectory + '...', flush = True)		
		super().__init__(str(pathlib.Path(worm_subdirectory).parent), working_directory, validated_directory, print_prefix = '\t')		
		self.worm_times = [a_time for a_time in self.timepoints if os.path.isfile(human_subdirectory + os.path.sep + a_time + ' ' + 'bf.png')]
		self.temporal_radius = 10
		self.worm_subdirectory = worm_subdirectory	
		self.human_subdirectory = human_subdirectory
		self.worm_name = self.worm_subdirectory.replace(self.experiment_directory + os.path.sep, '')
		experiment_name = self.worm_subdirectory.split(os.path.sep)[-2].replace(' Run ', ' ')
		self.full_worm_name = experiment_name + ' ' + self.worm_name
		self.write_directory = self.working_directory + os.path.sep + self.full_worm_name
		folderStuff.ensure_folder(self.write_directory)
		with open(worm_subdirectory + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
			self.position_metadata = json.loads(read_file.read())
		shutil.copyfile(worm_subdirectory + os.path.sep + 'position_metadata_extended.json', self.write_directory + os.path.sep + 'position_metadata_extended.json')
		self.bacterial_lawn = freeimage.read(worm_subdirectory + os.path.sep + 'great_lawn.png').astype('bool')
		self.worm_frame = pd.DataFrame(index = self.worm_times, columns = [
			'intensity_50', 'intensity_60', 'intensity_70', 'intensity_80', 'intensity_90', 'intensity_95', 'intensity_100', 'integrated_50', 'integrated_60', 'integrated_70', 'integrated_80', 'integrated_90', 'integrated_95', 'integrated_0', 
			'age_texture', 'vulval_texture', 'egg_texture', 'life_texture', 'intensity_texture',
			'unstimulated_rate', 'stimulated_rate_a', 'stimulated_rate_b', 'bulk_movement', 'fine_movement',
			'total_size', 'aspect_ratio',
			'visible_area', 'visible_eggs', 'average_egg_size', 'single_eggs'
		])	
		
	def read_corrected_bf(self, time_point, movement_key = ''):
		'''
		Read in an image at time_point and properly correct it for flatfield and metering.
		'''
		raw_image = freeimage.read(self.worm_subdirectory + os.path.sep + time_point + ' ' + 'bf' + movement_key + '.png')		
		flatfield_image = freeimage.read(self.calibration_directory + os.path.sep + time_point + ' ' + 'bf_flatfield.tiff')
		time_reference = self.metadata['brightfield metering'][time_point]['ref_intensity']
		raw_image[np.invert(self.super_vignette)] = 0
		corrected_image = raw_image*flatfield_image	
		corrected_image = corrected_image / time_reference * self.bf_reference_intensity
		corrected_image = corrected_image.astype('uint16')
		return corrected_image	
	
	def read_corrected_fluorescence(self, time_point, hot_threshold = 500):
		'''
		Correct fluorescence images for flatfield, and re-normalize to make the images more nicely viewable.	
		'''
		# Read in image and apply vignetting.
		image_path = self.worm_subdirectory + os.path.sep + time_point + ' ' + 'green_yellow_excitation_autofluorescence.png'
		raw_image = freeimage.read(image_path)
		raw_image[np.invert(self.super_vignette)] = 0

		# Correct for flatfield.
		flatfield_path = self.calibration_directory + os.path.sep + time_point + ' ' + 'fl_flatfield.tiff'
		calibration_image = freeimage.read(flatfield_path)
		corrected_image = raw_image*calibration_image	

		# Correct for hot pixels.
		median_image = scipy.ndimage.filters.median_filter(corrected_image, size = 3)
		difference_image = np.abs(corrected_image.astype('float64') - median_image.astype('float64')).astype('uint16')
		hot_pixels = difference_image > hot_threshold
		median_image_hot_pixels = median_image[hot_pixels]
		corrected_image[hot_pixels] = median_image_hot_pixels

		# Return the actual image.
		return corrected_image
	
	def write_results(self, my_time, focal_frame, background_frame, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored, write_health = True):
		'''
		Write out my results to disk as I go. This is how things should go!
		'''
		# Prepare directories and a base name to write to for the time point.
		os.makedirs(self.working_directory, exist_ok = True)
		os.makedirs(self.write_directory, exist_ok = True)
		base_name = self.write_directory + os.path.sep + my_time + ' '

		# Write out my images.
		freeimage.write(focal_frame.astype('uint16'), base_name + 'bf.png')
		freeimage.write(imageOperations.renormalize_image(focal_mask.astype('uint8')), base_name + 'mask.png')
		freeimage.write(imageOperations.renormalize_image(focal_fluorescence.astype('uint16')), base_name + 'fluorescence.png')
		freeimage.write(background_frame, base_name + 'background.png')
		freeimage.write(worm_eggs_colored, base_name + 'color_worm_eggs.png')
		freeimage.write(fluorescence_colored, base_name + 'color_fluorescence.png')
		freeimage.write(movement_colored, base_name + 'color_movement.png')
		freeimage.write(bulk_move_colored, base_name + 'color_bulk_movement.png')
		
		# Write out the measured data.
		if write_health:
			final_directory = self.experiment_directory + os.path.sep + 'measured_health'
			os.makedirs(final_directory, exist_ok = True)
			self.worm_frame.to_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t')
		return

	def add_metainformation(self):
		'''
		Add information about worm age, birth, death, and maturity to our final measured health dataframe.
		'''
		# Ensure that my output directory exists.
		final_directory = self.experiment_directory + os.path.sep + 'measured_health'
		os.makedirs(final_directory, exist_ok = True)
		
		# Add in the important elements!
		self.worm_frame = pd.read_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t', index_col = 0)
		for a_time in self.worm_frame.index:
			i = self.timepoints.index(a_time)
			self.worm_frame.loc[self.timepoints[i], 'age'] = self.position_metadata[i]['age']
			self.worm_frame.loc[self.timepoints[i], 'egg_age'] = self.position_metadata[i]['egg_age']
			self.worm_frame.loc[self.timepoints[i], 'ghost_age'] = self.position_metadata[i]['ghost_age']
		self.worm_frame.to_csv(final_directory + os.path.sep + self.worm_name + '.tsv', sep = '\t')
		return
	
	def mask_decision(self, my_time, focal_frame, print_suffix = ''):
		'''
		Generate a final mask based on metadata information and my three masks.
		'''		
		# Compute edge masks.
		self.t0 = time.time()
		(eggs_mask, edge_mask, my_edges) = edgeMorphology.find_eggs_worm(focal_frame, self.super_vignette, self.bacterial_lawn, worm_sigma = 2.0)
		print('\tComputed edge mask for ' + my_time + print_suffix + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		focal_mask = freeimage.read(self.human_subdirectory + os.path.sep + my_time + ' hmask.png').astype('bool')
		print('\tMade final decision for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		return (focal_mask, eggs_mask)
		
	def make_measurements(self, my_time, last_mask, focal_mask, movement_masks, last_frame, focal_frame, eggs_mask):
		'''
		Given the positions of the worm and eggs and raw data files, make the actual measurements that I want.
		'''
		# Measure movement, one way or another.
		self.t0 = time.time()			
		j = self.timepoints.index(my_time)
		(self.worm_frame.loc[my_time], movement_colored, bulk_move_colored) = extractFeatures.measure_movementCOM(movement_masks, [last_mask, focal_mask], [last_frame, focal_frame], self.position_metadata[j], self.worm_frame.loc[my_time]) 
				
		# Measure size and set up last_mask.
		last_mask = focal_mask.copy()		
		last_frame = focal_frame.copy()
		self.worm_frame.loc[my_time] = extractFeatures.measure_size(focal_mask, self.worm_frame.loc[my_time])
		
		# Only do autofluorescence measurements if fluorescence was on for this time point.
		if os.path.isfile(self.worm_subdirectory + os.path.sep + my_time + ' ' + 'green_yellow_excitation_autofluorescence.png'):
			focal_fluorescence = self.read_corrected_fluorescence(my_time)
			(self.worm_frame.loc[my_time], fluorescence_colored) = extractFeatures.measure_autofluorescence(focal_fluorescence, focal_mask, self.worm_frame.loc[my_time])
		else:
			focal_fluorescence = np.zeros(focal_frame.shape).astype('uint8')
			fluorescence_colored = focal_fluorescence
		print('\tMeasured movement, size, and fluorescence for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Measure the various textures.
		self.t0 = time.time()			
		self.worm_frame.loc[my_time] = extractFeatures.measure_texture(focal_frame, focal_mask, [self.age_texture_codebook, self.egg_texture_codebook, self.ghost_texture_codebook], [self.age_texture_regressor, self.egg_texture_regressor, self.life_texture_regressor], self.worm_frame.loc[my_time])
		print('\tMeasured texture for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Measure the eggs.
		self.t0 = time.time()			
		(self.worm_frame.loc[my_time], worm_eggs_colored) = extractFeatures.measure_eggs(eggs_mask, focal_mask, self.worm_frame.loc[my_time])
		print('\tMeasured eggs for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)
		return (last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored)
		
	def measure_a_time(self, my_time, last_frame, last_mask):
		'''
		Measure a time point for this worm. It is called by self.measure_a_worm().
		'''
		i = self.worm_times.index(my_time)
		print(str(i+1) + '/' + str(self.max_legitimate_times) + ': Measuring ' + my_time + ', ' + self.full_worm_name + '.', flush = True)

		# Find the worm!!! This part is different between measure_one_time and measure_a_time.			
		self.t1 = time.time()
		self.t0 = time.time()
		last_index =  max(i - 1, 0)
		last_time = self.worm_times[last_index]
		last_frame = self.read_corrected_bf(last_time)
		last_mask = freeimage.read(self.human_subdirectory + os.path.sep + last_time + ' ' + 'hmask.png').astype('bool')
		focal_frame = self.read_corrected_bf(my_time)
		movement_frames = [self.read_corrected_bf(my_time, movement_key = movement_key) for movement_key in ['00', '01', '10', '11', '12']]
		background_model = freeimage.read(self.working_directory + os.path.sep + self.full_worm_name + os.path.sep + last_time + ' ' + 'background.png')

		# Write out my background subtraction masks.
		(background_mask, movement_masks) = backgroundSubtraction.subtract_one_time(background_model, focal_frame, movement_frames, self.bacterial_lawn)
		print('\tGot masks from background subtraction for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)

		# Find the worm!!! This part is the same for both measure_one_time and measure_a_time.
		(focal_mask, eggs_mask) = self.mask_decision(my_time, focal_frame)	
		print('\tGot final masks for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t1)[:5] + ' seconds.', flush = True)

		# Make all my measurements.
		(last_frame, last_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored) = self.make_measurements(my_time, last_mask, focal_mask, movement_masks, last_frame, focal_frame, eggs_mask)		

		self.last_size = min(self.worm_frame.loc[my_time, 'total_size'], 100000)
		
		# Write out my results.
		self.t0 = time.time()	
		self.write_results(my_time, focal_frame, background_model, focal_mask, focal_fluorescence, worm_eggs_colored, fluorescence_colored, movement_colored, bulk_move_colored)
		print('\tWrote results for ' + my_time + ', ' + self.full_worm_name + ', took ' +  str(time.time()-self.t0)[:5] + ' seconds.', flush = True)		
		return

	def measure_a_worm_try(self):
		'''
		Try to fully measure a worm! If it fails, throw a fit so you get noticed!!! 
		'''	
		try:
			self.measure_a_worm()
		except:
			while True:
				print(self.full_worm_name + ' HAS FAILED!!!!!!!!!!!! PANIC!!!!!!!!!!!!!!!!!!!!', flush = True)
		return
	
	def measure_a_worm(self):
		'''
		Fully measure a worm!
		'''	
		# Run background subtraction forward.
		self.last_size = 1
		death_frame = [i for i in range(0, len(self.position_metadata)) if abs(self.position_metadata[i]['ghost_age']) < 0.1][0] + 1
		self.max_legitimate_times = min(len(list(self.worm_frame.index)), len(self.position_metadata), death_frame)
		print('Starting to measure ' + self.full_worm_name + '.', flush = True)
		last_mask = None
		last_frame = None
		for i in range(0, self.max_legitimate_times):
			self.measure_a_time(self.worm_times[i], last_frame, last_mask)
		self.add_metainformation()
		return


class HumanCheckpoints():
	'''
	A class to help with human-picking checkpoints for worm_directories.
	'''
	def __init__(self, data_directory, more_directories = {}, annotation_file = None):
		self.data_directory = data_directory
		self.more_directories = more_directories
		if annotation_file == None:
			if more_directories == {}:
				dir_files = os.listdir(data_directory)
				dir_files = [a_file for a_file in dir_files if a_file.split('.')[-1] == 'tsv']
				dir_files.append('DummyFile')
				try_file = data_directory + os.path.sep + dir_files[0]
				if os.path.isfile(try_file):
					annotation_file = try_file
			else:
				dir_files = os.listdir(more_directories['W'])
				dir_files = [a_file for a_file in dir_files if a_file.split('.')[-1] == 'tsv']
				dir_files.append('DummyFile')
				try_file = more_directories['W'] + os.path.sep + dir_files[0]
				if os.path.isfile(try_file):
					annotation_file = try_file				
		self.read_annotations(annotation_file)
	
		self.ris_made = False
		gc.collect()
		self.worm_directories = sorted([data_directory + os.path.sep + an_item for an_item in os.listdir(data_directory) if os.path.isdir(data_directory + os.path.sep + an_item) and an_item[0] in '0123456789'])
		self.worms = [worm_directory.replace(data_directory + os.path.sep, '') for worm_directory in self.worm_directories]
		self.load_arguments = {worm_directory.split(os.path.sep)[-1]: bright_field_paths(worm_directory) for worm_directory in self.worm_directories}
		self.checkpoints = pd.DataFrame(index = self.worms, columns = ['Hatch', 'Egg_Maturity', 'Death', 'Hatch_Prefix', 'Egg_Maturity_Prefix', 'Death_Prefix'])
		with open(data_directory + os.path.sep + 'experiment_metadata.json', 'r') as read_file:
			self.metadata = json.loads(read_file.read())
		self.timepoints = self.metadata['timepoints']
		self.timepoints_dict = {}
		for a_dir in sorted(list(self.more_directories.keys())):
			with open(more_directories[a_dir] + os.path.sep + 'experiment_metadata.json', 'r') as read_file:
				dir_metadata = json.loads(read_file.read())
			self.timepoints_dict[a_dir] = dir_metadata['timepoints']
	
	def load_recents(self, a_time = -1, only_living = False):
		'''
		Load the most recent bright field image.
		'''
		if not self.ris_made:
			global gui_called
			if not gui_called:
				gui_answer = input('Have you called \'%gui qt5?\n\t')
				if gui_answer.lower() == 'y':
					gui_called = True
				else:
					raise BaseException('Call \'%gui qt5 first!')
			from ris_widget.ris_widget import RisWidget
			self.ris_widget = RisWidget()
			self.ris_made = True
		self.ris_widget.show()
		load_arguments = one_time_paths(self.data_directory, self.timepoints[a_time])
		if only_living:
			load_arguments[0] = [a_file for a_file in load_arguments[0] if a_file.split(os.path.sep)[-2] in self.living_worms]
		self.ris_widget.main_flipbook.pages = []
		print('\tCleaned up ' + str(gc.collect()) + ' things.')
		self.ris_widget.main_flipbook._handle_dropped_files(*load_arguments)		
		return

	def willie_load(self, index, my_range = (None, None)):
		print('Loading images for the worm ' + str(index).zfill(3) + '.')
		# Convert the given range into a mutable list.
		my_range = list(my_range)
		
		# Check to make sure that we're able to make the ris_widget gui.
		if not self.ris_made:
			global gui_called
			if not gui_called:
				gui_answer = input('Have you called \'%gui qt5\'?\n\t')
				if gui_answer.lower() == 'y':
					gui_called = True
				else:
					raise BaseException('Call \'%gui qt5\' first!')
			from ris_widget.ris_widget import RisWidget
			self.ris_widget = RisWidget()
			self.ris_made = True
		self.ris_widget.show()
		self.ris_widget.main_flipbook.pages = []
		
		# Garbage collect and report results.
		print('\tCleaned up ' + str(gc.collect()) + ' things.')
		
		# Check different possible string names for the given index.
		minimum_z = len(str(index))
		for i in range(minimum_z, 4):
			if str(index).zfill(i) in self.worms:
				# Load only the specified range of images, but do it in the correct place.
				my_load = list(self.load_arguments[str(index).zfill(i)])
				new_load = list(my_load[0])
				if my_range[0] == None:
					my_range[0] = 0
				elif my_range[0] < 0:
					my_range[0] = len(new_load) + my_range[0]
				if my_range[1] == None:
					my_range[1] = len(new_load)
				elif my_range[1] < 0:
					my_range[1] = len(new_load) + my_range[1]

				for i in range(0, my_range[0]):
					new_load[i] = 'INDEXING PLACEHOLDER'
				for j in range(my_range[1], len(new_load)):
					new_load[j] = 'INDEXING PLACEHOLDER'
				my_load[0] = new_load
				self.ris_widget.main_flipbook._handle_dropped_files(*my_load)
		return
	
	def parse_drying(self):
		'''
		Learn about the drying curves.
		'''
		drying_file = self.data_directory + os.path.sep + 'acquisitions.log'
		with open(drying_file, 'r') as read_file:
			drying_data = read_file.read()
		drying_lines = drying_data.split('\n')
		drying_lines = [a_line.split('\t') for a_line in drying_lines]
		position_zs = {}
		for i in range(0, len(drying_lines)):
			a_line = drying_lines[i]
			if a_line[-1][:20] == 'Acquiring Position: ':
				my_well = a_line[-1].split(' ')[-1]
				next_line = drying_lines[i+1]
				z_pos = float(next_line[-1].split(' ')[-1])
				if my_well not in position_zs.keys():
					position_zs[my_well] = [z_pos]
				else:
					position_zs[my_well].append(z_pos)				
		return position_zs		
	
	def worm_name(self, worm_directory):
		'''
		Returns the worm name portion of a worm directory.
		'''
		return worm_directory.replace(self.data_directory + os.path.sep, '')
		
	def record_checkpoints(self, worm_name_or_index):
		'''
		Record checkpoints for a worm.
		'''
		if type(worm_name_or_index) == type(''):
			worm_name = worm_name_or_index
		elif type(worm_name_or_index) == type(0):
			worm_name = self.worms[worm_name_or_index]
		if worm_name not in self.worms:
			raise BaseException('Invalid worm name or index.')
		
		my_indices = input('Please enter the indices for ' + worm_name + ' as Hatch Vulval_Maturity Egg_Maturity Death:\n\t')
		my_indices = [int(an_index.strip()) for an_index in my_indices.split(' ')]
		my_indices = [self.timepoints[i] for i in my_indices]
		self.checkpoints.loc[worm_name, :] = my_indices
		return
		
	def find_positions(self, worm_names):
		'''
		Find positions of a given worm or "ref" for reference positions.
		'''
		if worm_names == 'ref':
			my_positions = self.metadata['reference_positions']
		else:			
			my_positions = [self.metadata['positions'][a_worm] for a_worm in worm_names]
		return np.array(my_positions)

	def record_from_frame(self, checkpoint_df):
		'''
		Record information from a filled out dataframe.
		'''
		for worm_name in self.checkpoints.index:
			my_index = checkpoint_df[checkpoint_df.loc[:, 'Worm_Index'] == worm_name].index[0]
			my_indices = list(checkpoint_df.loc[my_index, ['Hatch', 'First Egg Laid', 'Death']])
			new_indices = []
			new_prefixes = []
			for an_index in my_indices:
				an_index = str(an_index)
				index_prefix = an_index[0]
				if index_prefix.isdigit():
					my_time = self.timepoints[int(float(an_index))]
				else:
					my_time = self.timepoints_dict[index_prefix][int(float(an_index[1:]))]
				new_prefixes.append(index_prefix)
				new_indices.append(my_time)
			self.checkpoints.loc[worm_name, ['Hatch', 'Egg_Maturity', 'Death']] = new_indices
			self.checkpoints.loc[worm_name, ['Hatch_Prefix', 'Egg_Maturity_Prefix', 'Death_Prefix']] = new_prefixes
		return

	def get_timestamp(self, index_prefix, an_index, position_metadata, timestamp_dict, position_metadata_dict, timestamp_dicts_dict):			
		'''
		Get a timestamp...
		'''
		if index_prefix.isdigit():
			my_stamp = timestamp_dict[an_index]
		else:
			my_stamp = timestamp_dicts_dict[index_prefix][an_index]			
		return my_stamp

	def get_timestamp_i(self, my_index, position_metadata, timestamp_dict, position_metadata_dict, timestamp_dicts_dict):	
		'''
		Get a timestamp using a raw index instead of the timepoint name.
		'''
		my_index = str(my_index)
		index_prefix = my_index[0]
		if index_prefix.isdigit():
			an_index = int(float(my_index))
			my_stamp = timestamp_dict[self.timepoints[an_index]]
		else:
			an_index = int(float(my_index[1:]))
			my_stamp = timestamp_dicts_dict[index_prefix][self.timepoints_dict[index_prefix][an_index]]			
		return my_stamp

	def make_drew_tsv(self, annotation_file):
		'''
		Make a .tsv of important times and information for Drew.
		'''
		# Read in the annotations and fill in some non-numerical values.
		latest_annotations = pd.read_csv(annotation_file, sep = '\t')
		latest_annotations = latest_annotations.replace(np.nan, 0)
		latest_annotations = latest_annotations.replace('X', 0)
		latest_annotations.loc[:, 'Worm'] = [a_worm.split('/')[-1] for a_worm in latest_annotations.loc[:, 'Worm']] 
		latest_annotations.index = latest_annotations.loc[:, 'Worm']
		timestamp_df = pd.DataFrame(index = latest_annotations.index, columns = latest_annotations.columns)
		timestamp_df.loc[:, ['Worm', 'Notes']] = latest_annotations.loc[:, ['Worm', 'Notes']]

		for worm_directory in self.worm_directories:
			worm_name = self.worm_name(worm_directory)
			with open(worm_directory + os.path.sep + 'position_metadata.json', 'r') as read_file:
				position_metadata = json.loads(read_file.read())
			timestamp_dict = {self.timepoints[i]: position_metadata[i]['timestamp'] for i in range(0, len(position_metadata))}
			position_metadata_dict = {}
			timestamp_dicts_dict = {}
			for a_dir in sorted(list(self.more_directories.keys())):
				with open(self.more_directories[a_dir] + os.path.sep + worm_name + os.path.sep + 'position_metadata.json', 'r') as read_file:
					more_position_metadata = json.loads(read_file.read())
				position_metadata_dict[a_dir] = more_position_metadata
				timestamp_dicts_dict[a_dir] = {self.timepoints_dict[a_dir][i]: more_position_metadata[i]['timestamp'] for i in range(0, len(more_position_metadata))}

			for a_column in latest_annotations.columns:
				if a_column not in ['Worm', 'Notes']:
					timestamp_df.loc[worm_name, a_column] = self.get_timestamp_i(latest_annotations.loc[worm_name, a_column], position_metadata, timestamp_dict, position_metadata_dict, timestamp_dicts_dict)				
#		timestamp_df.to_csv(self.data_directory + os.path.sep + 'drew.tsv', sep = '\t')
		return

	def save_checkpoints(self, experiment_directory = None):
		'''
		Save checkpoints for a worm.
		'''
		if experiment_directory == None or experiment_directory.isdigit():
			my_prefix = '0'			
			experiment_directory = self.data_directory
		else: 
			my_prefix = experiment_directory
			experiment_directory = self.more_directories[experiment_directory]

		for worm_directory in self.worm_directories:
			worm_name = self.worm_name(worm_directory)
			if os.path.isfile(experiment_directory + os.path.sep + worm_name + os.path.sep + 'position_metadata.json'):
				with open(worm_directory + os.path.sep + 'position_metadata.json', 'r') as read_file:
					position_metadata = json.loads(read_file.read())
				timestamp_dict = {self.timepoints[i]: position_metadata[i]['timestamp'] for i in range(0, len(position_metadata))}
				position_metadata_dict = {}
				timestamp_dicts_dict = {}
				for a_dir in sorted(list(self.more_directories.keys())):
					with open(self.more_directories[a_dir] + os.path.sep + worm_name + os.path.sep + 'position_metadata.json', 'r') as read_file:
						more_position_metadata = json.loads(read_file.read())
					position_metadata_dict[a_dir] = more_position_metadata
					timestamp_dicts_dict[a_dir] = {self.timepoints_dict[a_dir][i]: more_position_metadata[i]['timestamp'] for i in range(0, len(more_position_metadata))}
	
				with open(experiment_directory + os.path.sep + worm_name + os.path.sep + 'position_metadata.json', 'r') as read_file:
					my_position_metadata = json.loads(read_file.read())
	
				for i in range(0, len(my_position_metadata)):
					now_time = my_position_metadata[i]['timestamp']
					seconds_age = now_time - self.get_timestamp(self.checkpoints.loc[worm_name, 'Hatch_Prefix'], self.checkpoints.loc[worm_name, 'Hatch'], position_metadata, timestamp_dict, position_metadata_dict, timestamp_dicts_dict)
					seconds_egg_age = now_time - self.get_timestamp(self.checkpoints.loc[worm_name, 'Egg_Maturity_Prefix'], self.checkpoints.loc[worm_name, 'Egg_Maturity'], position_metadata, timestamp_dict, position_metadata_dict, timestamp_dicts_dict)
					seconds_ghost_age = now_time - self.get_timestamp(self.checkpoints.loc[worm_name, 'Death_Prefix'], self.checkpoints.loc[worm_name, 'Death'], position_metadata, timestamp_dict, position_metadata_dict, timestamp_dicts_dict)
	
					my_position_metadata[i]['age'] = seconds_age/3600
					my_position_metadata[i]['egg_age'] = seconds_egg_age/3600
					my_position_metadata[i]['ghost_age'] = seconds_ghost_age/3600
	
					my_position_metadata[i]['hatch'] = self.checkpoints.loc[worm_name, 'Hatch']
					my_position_metadata[i]['egg_maturity'] = self.checkpoints.loc[worm_name, 'Egg_Maturity']
					my_position_metadata[i]['death'] = self.checkpoints.loc[worm_name, 'Death']
					if my_prefix.isdigit():
						my_position_metadata[i]['timepoint'] = self.timepoints[i]
					else:
						my_position_metadata[i]['timepoint'] = self.timepoints_dict[my_prefix][i]	
				with open(experiment_directory + os.path.sep + worm_name + os.path.sep + 'position_metadata_extended.json', 'w') as write_file:
					write_file.write(json.dumps(my_position_metadata, sort_keys=True, indent=4))
		return
		
	def show_worms(self):
		'''
		Show my worms.
		'''
		reference_positions = self.find_positions('ref')
		bad_positions = self.find_positions(self.bad_worms)
		good_positions = self.find_positions(self.good_worms)
		living_positions = self.find_positions(self.living_worms)
		plt.figure()
		plt.scatter(bad_positions[:, 0], bad_positions[:, 1], color = 'red')
		plt.scatter(good_positions[:, 0], good_positions[:, 1], color = 'blue')
		if len(living_positions > 0):
			plt.scatter(living_positions[:, 0], living_positions[:, 1], color = 'green')
		plt.scatter(reference_positions[:, 0], reference_positions[:, 1], color = 'black')
		plt.show()		
		return
		
	def delete_bad_wells(self, all_delete = False):
		'''
		Delete bad wells.
		'''
		if all_delete:
			my_input = input('Delete ' + self.data_directory + os.path.sep + str(self.bad_worms) + '? \nGood worms are ' + str(self.good_worms) + ' (Y/N)\n')
			if my_input.lower() == 'y':
				for a_worm in self.bad_worms:
					if os.path.exists(self.data_directory + os.path.sep + a_worm):
						shutil.rmtree(self.data_directory + os.path.sep + a_worm)
		else:
			for a_worm in self.bad_worms:
				if a_worm in self.good_worms:
					raise('AAARGH')
				if os.path.exists(self.data_directory + os.path.sep + a_worm):
					my_input = input('Delete ' + self.data_directory + os.path.sep + a_worm + '? (Y/N)\n')
					if my_input.lower() == 'y':
						shutil.rmtree(self.data_directory + os.path.sep + a_worm)
					else:
						pass
		return

	def select_validations(self, complete_df):
		'''
		Select time points/worms to validate.
		'''
		egg_worms = sorted(list(np.random.choice(list(complete_df.raw.keys()), 35, replace = False)))
		
		
		all_times = []
		for a_worm in list(complete_df.raw.keys()):
		    my_times = list(complete_df.raw[a_worm][(complete_df.raw[a_worm].loc[:, 'egg_age'] >= 0) & (complete_df.raw[a_worm].loc[:, 'ghost_age'] <= 0)].index)
		    all_times.extend([a_worm + os.path.sep + my_time for my_time in my_times])
		egg_counts = sorted(list(set(list(np.random.choice(all_times, 150, replace = False)))))
		
		age_bins = [(i, i+72) for i in np.array(list(range(10)))*72]
		bin_times = [[] for i in range(0, len(age_bins))]
		for i in range(0, len(age_bins)):
			a_bin = age_bins[i]
			for a_worm in list(complete_df.raw.keys())[:21]:
				my_times = list(complete_df.raw[a_worm][(complete_df.raw[a_worm].loc[:, 'age'] >= a_bin[0]) & (complete_df.raw[a_worm].loc[:, 'ghost_age'] <= 0) & (complete_df.raw[a_worm].loc[:, 'age'] <= a_bin[1])].index)
				bin_times[i].extend([a_worm + os.path.sep + my_time for my_time in my_times])
		
		chosen_masks = []
		for a_bin in bin_times:
			chosen_ones = np.random.choice(a_bin, 40)
			chosen_masks.append(chosen_ones)		
		return (chosen_masks, egg_counts, egg_worms)
	
	def read_annotations(self, annotation_file):
		'''
		Read in my annotation file data.
		'''
		if annotation_file != None:
			# Read in the annotations and fill in some non-numerical values.
			latest_annotations = pd.read_csv(annotation_file, sep = '\t')
			latest_annotations = latest_annotations.replace(np.nan, 0)
			latest_annotations = latest_annotations.replace('X', 0)
			
			# Figure out which worms are alive/dead/bad wells based on the "Notes" column.
			latest_annotations.loc[:, 'Worm_Index'] = [str(weird_name).split('/')[-1] for weird_name in latest_annotations.loc[:, 'Worm']]
			latest_annotations.loc[:, 'Good_Worm'] = pd.Series(['DEAD' in str(a_note) for a_note in latest_annotations.loc[:, 'Notes']])
			latest_annotations.loc[:, 'Dead'] = pd.Series(['DEAD' in str(a_note) and 'NOT DEAD' not in str(a_note) for a_note in latest_annotations.loc[:, 'Notes']])
			self.bad_worms = list(latest_annotations[latest_annotations.loc[:, 'Good_Worm'] == False].loc[:, 'Worm_Index'])
			self.good_worms = list(latest_annotations[latest_annotations.loc[:, 'Good_Worm']].loc[:, 'Worm_Index'])
			self.dead_worms = list(latest_annotations[latest_annotations.loc[:, 'Dead']].loc[:, 'Worm_Index'])
			self.living_worms = [a_worm for a_worm in self.good_worms if a_worm not in self.dead_worms]
			self.all_worms = [a_worm for a_worm in latest_annotations.loc[:, 'Worm_Index']]
			self.latest_annotations = latest_annotations
		else:
			self.bad_worms = None
			self.good_worms = None
			self.dead_worms = None
			self.living_worms = None
			self.all_worms = None
			self.latest_annotations = None
		return
	
	def clean_experiment_directory(self, annotation_file = None, experiment_directory = None):
		'''
		Read in an annotation_file and use its information to remove bad worms from an experiment and start skipping the dead ones.
		'''
		if experiment_directory == None:
			my_prefix = '0'
			experiment_directory = self.data_directory
		else:
			my_prefix = experiment_directory
			experiment_directory = self.more_directories[experiment_directory]
		
		# Figure out a list of wells to skip acquisitions on.
		skip_list = list(self.bad_worms)
		skip_list.extend(self.dead_worms)
		skip_string = str(sorted(skip_list)).replace('\'', '\"')
		print('Skip wells: ' + skip_string, flush = True)		
		
		# Ensure that you're not skipping any worms we still want.
		for a_worm in self.living_worms:
			if a_worm in skip_list:
				raise BaseException('Skipping a living worm!')
		for a_worm in self.all_worms:
			if a_worm not in self.living_worms and a_worm not in skip_list:
				raise BaseException('Missing a worm!')
		if len(self.living_worms) + len(skip_list) != len(self.all_worms):
			raise BaseException('The worms are not partitioned properly. They don\'t add up.')
		
		# Save out our metadata.
		self.record_from_frame(self.latest_annotations)
		self.save_checkpoints(my_prefix)
		self.move_dead(my_prefix)
#		self.make_drew_tsv(annotation_file)
		return


	def delete_dead(self, experiment_directory = None):
		'''
		Delete frames after death from the subdirectory to save space.
		'''
		if experiment_directory == None or experiment_directory.isdigit():
			my_prefix = '0'			
			experiment_directory = self.data_directory
		else: 
			my_prefix = experiment_directory
			experiment_directory = self.more_directories[my_prefix]
		
		for worm_directory in self.worm_directories:
			worm_name = self.worm_name(worm_directory)
			if worm_name in self.good_worms and worm_name in self.dead_worms:
				move_directory = experiment_directory + os.path.sep + worm_name
				shutil.rmtree(move_directory + os.path.sep + 'life_after_death')
		return

	def move_dead(self, experiment_directory):
		'''
		Move frames after death to a subdirectory, in case they are out of focus and cause problems for great_lawn detection.
		'''
		if experiment_directory == None or experiment_directory.isdigit():
			my_prefix = '0'			
			experiment_directory = self.data_directory
		else: 
			my_prefix = experiment_directory
			experiment_directory = self.more_directories[my_prefix]

		for worm_directory in self.worm_directories:
			worm_name = self.worm_name(worm_directory)
			if worm_name in self.good_worms and worm_name in self.dead_worms:
				move_directory = experiment_directory + os.path.sep + worm_name
				with open(move_directory + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
					position_metadata = json.loads(read_file.read())
				death_timepoint = [a_time for a_time in position_metadata if a_time['timepoint'] == a_time['death']][0]
				death_stamp = death_timepoint['timestamp']
				for a_timepoint in position_metadata:
					if a_timepoint['timestamp'] > death_stamp + 1:
						my_time = a_timepoint['timepoint']
						folderStuff.ensure_folder(move_directory + os.path.sep + 'life_after_death')
						for an_extension in ['bf', 'bf00', 'bf01', 'bf10', 'bf11', 'bf12', 'green_yellow_excitation_autofluorescence']:
							if os.path.isfile(move_directory + os.path.sep + my_time + ' ' + an_extension + '.png'):
								shutil.move(move_directory + os.path.sep + my_time + ' ' + an_extension + '.png', move_directory + os.path.sep + 'life_after_death' + os.path.sep + my_time + ' ' + an_extension + '.png')
		return

def human_measure(directory_bolus, parallel = True):
	'''
	Run human_a_worm for all worms in human_directory using training data from validated_directory and write the output to /mnt/bulkdata/wzhang/human_dir.	
	'''	
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	t_total = time.time()

	my_worm_list = [a_worm for a_worm in os.listdir(directory_bolus.human_directory) if os.path.isdir(directory_bolus.human_directory + os.path.sep + a_worm)]
	worm_data_dirs = []
	for a_worm in my_worm_list:
		split_worm = a_worm.split(' ')
		search_base_dir = ' '.join(split_worm[:-2]) + ' Run ' + split_worm[-2]
		worm_data_dir = [a_dir for a_dir in directory_bolus.data_directories if search_base_dir in a_dir][0]
		worm_data_dir += os.path.sep + split_worm[-1]
		worm_data_dirs.append(worm_data_dir)
	my_worm_list = [directory_bolus.human_directory + os.path.sep + a_worm for a_worm in my_worm_list]

	# Make base measurers.
	my_bases = [BaseMeasurer(directory_bolus.data_directories[i], directory_bolus.working_directory, directory_bolus.human_directory) for i in range(directory_bolus.done, directory_bolus.ready)]

	# Make worm measurers.		
	my_worms = {}
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		worm_measure_list = [executor.submit(HumanWormMeasurer, worm_data_dirs[j], my_worm_list[j], directory_bolus.working_directory, directory_bolus.human_directory) for j in range(0, len(my_worm_list))]
	concurrent.futures.wait(worm_measure_list)
	worm_measure_list = [a_job.result() for a_job in worm_measure_list]
	for i in range(0, len(my_worm_list)):			
		my_worms[my_worm_list[i]] = worm_measure_list[i]

	# Measure my worms!
	if parallel:	
		with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
			my_measures = [executor.submit(my_worms[a_worm].measure_a_worm_try) for a_worm in sorted(list(my_worms.keys()))]
		# Wait for the results and combine them.
		concurrent.futures.wait(my_measures)
		my_measures = [a_job.result() for a_job in my_measures]
	# Measure my worms! (debug mode)
	else:
		for a_worm in my_worms.keys():
			my_worms[a_worm].measure_a_worm()
	print('Total time taken for measurements: ' + str(time.time() - t_total) + ' seconds.')
	return

def measure_experiments(directory_bolus, parallel = True, only_worm = None, only_time = None, refresh_metadata = False):
	'''
	Run measure_a_worm for all worms in data_directory using training data from validated_directory and write the output to working_directory.	
	'''	
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	t_total = time.time()

	# Read in all my metadata and make checker (HumanCheckpoints) objects.
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor: # This is parallel way.
		my_checkers = [executor.submit(make_checker, directory_bolus.data_directories[i], directory_bolus.extra_directories[i], directory_bolus.experiment_directories[i], directory_bolus.annotation_directories[i], refresh_metadata) for i in range(directory_bolus.done, directory_bolus.ready)]
	concurrent.futures.wait(my_checkers)
	my_checkers = [a_job.result() for a_job in my_checkers]
#	my_checkers = [make_checker(directory_bolus.data_directories[i], directory_bolus.extra_directories[i], directory_bolus.experiment_directories[i], directory_bolus.annotation_directories[i], refresh_metadata) for i in range(directory_bolus.done, directory_bolus.ready)] # This is slow way.

	# Make base measurers.
	my_bases = [BaseMeasurer(directory_bolus.data_directories[i], directory_bolus.working_directory, directory_bolus.human_directory) for i in range(directory_bolus.done, directory_bolus.ready)]

	# Make worm measurers.		
	my_worms = {}
	worm_list = []
	if only_worm == None:
		for i in range(0, len(my_checkers)):
			checker = my_checkers[i]
			my_base = my_bases[i]
			for worm_subdir in my_base.worm_dirs:
				if worm_subdir.split(os.path.sep)[-1] in checker.good_worms:
					worm_list.append(worm_subdir)					
		with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
			worm_measure_list = [executor.submit(WormMeasurer, worm_subdir, directory_bolus.working_directory, directory_bolus.human_directory) for worm_subdir in worm_list]
		concurrent.futures.wait(worm_measure_list)
		worm_measure_list = [a_job.result() for a_job in worm_measure_list]
		for i in range(0, len(worm_list)):			
			my_worms[worm_list[i]] = worm_measure_list[i]
	else:
		my_worms[only_worm] = WormMeasurer(only_worm, directory_bolus.working_directory, directory_bolus.human_directory)

	# Measure my worms!
	if parallel:	
		with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
			my_measures = [executor.submit(my_worms[a_worm].measure_a_worm_try) for a_worm in sorted(list(my_worms.keys()))]
		# Wait for the results and combine them.
		concurrent.futures.wait(my_measures)
		my_measures = [a_job.result() for a_job in my_measures]
	# Measure my worms! (debug mode)
	else:
		go_worms = []
		if only_worm == None:
			go_worms = sorted(list(my_worms.keys()))
		else:
			go_worms = [only_worm]
		if only_time == None:
			for a_worm in go_worms:
				my_worms[a_worm].measure_a_worm()
		else:
			my_worms[only_worm].measure_one_time(only_time)
	
	print('Total time taken for measurements: ' + str(time.time() - t_total) + ' seconds.')
	
	if parallel:	
		my_sizes = validate_generated_masks(directory_bolus.working_directory, directory_bolus.human_directory)
		check_worst_residuals(my_sizes)
	return

def make_checker(data_directory, extra_directories, experiment_directory, annotation_directory, refresh_metadata):
	'''
	Make checkers and clean up metadata if needed.
	'''
	# Figure out the metadata.
	if extra_directories != None:	
		checker = HumanCheckpoints(annotation_directory, extra_directories)
	else:
		checker = HumanCheckpoints(data_directory)
	if refresh_metadata:
		checker.clean_experiment_directory(experiment_directory = experiment_directory)	
	return checker

def validate_eggs(complete_df, directory_bolus, egg_directory = r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.05 spe-9 Human Eggs'):
	'''
	Check the correlation between my measured egg area and those of human-drawn masks.
	'''
	my_images = []
	for a_subdir in os.listdir(egg_directory):
		my_files = os.listdir(egg_directory + os.path.sep + a_subdir)
		for a_file in my_files:
			if a_file.split(' ')[-1] == 'emask.png':
				my_images.append(a_subdir + ' ' + a_file.split(' ')[0])

	eggs_df = pd.DataFrame(index = my_images, columns = ['H_Area', 'F_Area'])
	for an_image in eggs_df.index:
		split_name = an_image.split(' ')
		worm_name = ' '.join(split_name[:-1])
		image_file = egg_directory + os.path.sep + worm_name + os.path.sep + split_name[-1] + ' ' + 'emask.png'
		eggs_df.loc[an_image, 'F_Area'] = np.sum(freeimage.read(image_file).astype('bool'))
		eggs_df.loc[an_image, 'H_Area'] = complete_df.raw[worm_name].loc[split_name[-1], 'visible_area']
	return eggs_df

def validate_generated_masks(work_directory, human_directory, provided_df = None):
	'''
	Check the size correlation of my generated masks against the human masks.
	'''
	def good_name(a_mask_path):
		'''
		Make a nice name from a mask file path.
		'''
		my_chunks = a_mask_path.split(os.path.sep)[-2:]
		timepoint = my_chunks[-1].split(' ')[0]
		worm_name = my_chunks[0]
		my_name = '/'.join([worm_name, timepoint])
		return my_name
	
	if provided_df == None:
		my_human_masks = []
		for a_subdir in [a_dir for a_dir in os.listdir(human_directory) if os.path.isdir(human_directory + os.path.sep + a_dir)]:
			full_subdir = human_directory + os.path.sep + a_subdir
			for a_file in os.listdir(full_subdir):
				if a_file[0].isdigit():
					my_human_masks.append(full_subdir + os.path.sep + a_file.split(' ')[0])			
		my_human_masks = list(set(my_human_masks))
		my_human_masks = [a_mask + ' ' + 'hmask.png' for a_mask in my_human_masks]
		experiment_masks = [a_mask.replace(human_directory, work_directory).replace('hmask', 'mask') for a_mask in my_human_masks]
	
		my_sizes = pd.DataFrame(index = [good_name(a_mask) for a_mask in experiment_masks], columns = ['H_Size', 'F_Size', 'H_CentroidX', 'H_CentroidY', 'F_CentroidX', 'F_CentroidY'])
		for i in range(0, len(experiment_masks)):
			human_mask = freeimage.read(my_human_masks[i]).astype('bool')
			human_worm = np.array(np.where(human_mask > 0))
			human_center = np.mean(human_worm, axis = 1)
			experiment_mask = freeimage.read(experiment_masks[i]).astype('bool')
			experiment_worm = np.array(np.where(experiment_mask > 0))
			experiment_center = np.mean(experiment_worm, axis = 1)
			my_sizes.loc[good_name(experiment_masks[i]), 'H_Size'] = np.sum(human_mask)
			my_sizes.loc[good_name(experiment_masks[i]), 'F_Size'] = np.sum(experiment_mask)
			my_sizes.loc[good_name(experiment_masks[i]), 'H_CentroidX'] = human_center[0]
			my_sizes.loc[good_name(experiment_masks[i]), 'H_CentroidY'] = human_center[1]
			my_sizes.loc[good_name(experiment_masks[i]), 'F_CentroidX'] = experiment_center[0]
			my_sizes.loc[good_name(experiment_masks[i]), 'F_CentroidY'] = experiment_center[1]
	else:
		my_sizes = provided_df.copy()
	
	pearson_rsquared = scipy.stats.stats.pearsonr(list(my_sizes.loc[:, 'H_Size']), list(my_sizes.loc[:, 'F_Size']))[0]**2
	spearman_rsquared = scipy.stats.stats.spearmanr(list(my_sizes.loc[:, 'H_Size']), list(my_sizes.loc[:, 'F_Size']))[0]**2
	print('Pearson r^2 for measured worm size vs. human:', pearson_rsquared)
	print('Spearman r^2 for measured worm size vs. human:', spearman_rsquared)
	return my_sizes

def check_worst_residuals(size_df):
	'''
	Identify the animals and time points that yield the worst residuals between actual size and computed size.
	'''
	my_fit = np.polyfit(size_df.loc[:, 'F_Size'].values.astype('int64'), size_df.loc[:, 'H_Size'].values.astype('int64'), 1)
	fit_fn = np.poly1d(my_fit) 
	my_residuals = np.abs(size_df.loc[:, 'H_Size'] - fit_fn(size_df.loc[:, 'F_Size']))
	my_residuals.sort(ascending = False)
	return my_residuals.iloc[:10]

def organize_time_points(data_dir):
	'''
	Returns a list of all the time points in data_dir as strings.
	'''
	time_point_list = [a_file.split(' ')[0] for a_file in os.listdir(data_dir) if (a_file[-3:] == 'png' or a_file[-4:] == 'tiff') and a_file[0].isdigit()]
	time_point_list = list(set(time_point_list))
	return sorted(time_point_list)

def one_time_paths(a_directory, time_point, my_range = None):
	'''
	Give a list of all the paths to the bright field image at time_point for subdirectories in a_directory.

	View the images as follows:
		
		%gui qt5
		from ris_widget.ris_widget import RisWidget; rw = RisWidget(); rw.show()
		rw._main_flipbook._handle_dropped_files(*rw_load_arguments)
	'''
	my_subdirs = sorted([a_directory + os.path.sep + a_subdir for a_subdir in os.listdir(a_directory) if a_subdir[0] in '0123456789'])
	my_files = [a_subdir + os.path.sep + time_point + ' ' + 'bf.png' for a_subdir in my_subdirs if os.path.isfile(a_subdir + os.path.sep + time_point + ' ' + 'bf.png')]
	if len(my_files) == 0:
		my_files = [a_subdir + os.path.sep + time_point + ' ' + 'bf.tiff' for a_subdir in my_subdirs if os.path.isfile(a_subdir + os.path.sep + time_point + ' ' + 'bf.tiff')]
	if my_range == None:
		my_range = (0, len(my_files))
	if my_range[1] == 'end':
		my_range = (my_range[0], len(my_files))
	rw_load_arguments = [my_files[my_range[0]: my_range[1]], 0, 0, 0]	
	return rw_load_arguments

def bright_field_paths(a_subdirectory):
	'''
	Gets the names of all the cleaned bright field files in a_subdirectory in order, and returns their paths, along with filler optional arguments to send to ris_widget, which displays them after the follow are entered into IPython:
		
		%gui qt5
		from ris_widget.ris_widget import RisWidget; rw = RisWidget(); rw.show()
		rw._main_flipbook._handle_dropped_files(*rw_load_arguments)
	'''
	my_times = organize_time_points(a_subdirectory)
	my_paths = [a_subdirectory + os.path.sep + my_time + ' bf.png' for my_time in my_times if os.path.isfile(a_subdirectory + os.path.sep + my_time + ' bf.png')]
	if len(my_paths) == 0:
		my_paths = [a_subdirectory + os.path.sep + my_time + ' bf.tiff' for my_time in my_times if os.path.isfile(a_subdirectory + os.path.sep + my_time + ' bf.tiff')]
	rw_load_arguments = [my_paths, 0, 0, 0]
	return rw_load_arguments

def ordered_paths(a_subdirectory):
	'''
	Gets the names of all files in a_subdirectory in order, and returns their paths, along with filler optional arguments to send to ris_widget, which displays them after the follow are entered into IPython:
		
		%gui qt5
		from ris_widget.ris_widget import RisWidget; rw = RisWidget(); rw.show()
		rw._main_flipbook._handle_dropped_files(*rw_load_arguments)
	'''
	my_files = sorted(os.listdir(a_subdirectory))
	my_paths = [a_subdirectory + os.path.sep + my_file for my_file in my_files]
	rw_load_arguments = [my_paths, 0, 0, 0]
	return rw_load_arguments

def repeat_texture(adult_df, directory_bolus, parallel = False):
	'''
	Repeat my texture analysis only.
	'''
	# Make base measurers to get the texture classifier ready.	
	worm_list = sorted(list(adult_df.raw.keys()))
	my_base = BaseMeasurer(directory_bolus.data_directories[0], directory_bolus.working_directory, directory_bolus.human_directory)

	# Split up the job by worms and parallelize.
	if parallel:
		my_workers = min(multiprocessing.cpu_count() - 1, 60)
		with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
			new_raws = [executor.submit(repeat_texture_worm, directory_bolus.human_directory, directory_bolus.working_directory, directory_bolus.data_directories[0], a_worm, adult_df.raw[a_worm].copy()) for a_worm in worm_list]
		# Wait for the results and combine them.
		concurrent.futures.wait(new_raws)
		new_raws = [a_job.result() for a_job in new_raws]
		new_raws = {worm_list[i]:new_raws[i] for i in range(0, len(worm_list))}
	# Just use one core.
	else:
		new_raws = {}	
		for a_worm in worm_list:
			new_raws[a_worm] = repeat_texture_worm(directory_bolus.human_directory, directory_bolus.working_directory, directory_bolus.data_directories[0], a_worm, adult_df.raw[a_worm].copy())
			
	# Save out my results.
	for a_raw in new_raws.keys():
		if sys.platform == 'win32':
			new_raws[a_raw].to_csv(adult_df.save_directory + os.path.sep + a_raw + '.tsv', sep = '\t')
		else:
			folderStuff.ensure_folder(directory_bolus.human_directory + os.path.sep + 'measured_health_new')
			new_raws[a_raw].to_csv(directory_bolus.human_directory + os.path.sep + 'measured_health_new' + os.path.sep + a_raw + '.tsv', sep = '\t')		
	return

def repeat_texture_worm(human_directory, working_directory, data_directory, a_worm, a_frame):
	'''
	Repeat texture measurements for just one worm.
	'''
	# Get some stuff set up.	
	my_base = BaseMeasurer(data_directory, working_directory, human_directory)
	worm_times = sorted(list(a_frame.index))
	
	# Make the measurements.	
	print('Working on ' + a_worm + '.')
	for my_time in worm_times:
		print('\tWorking on ' + a_worm + ', time ' + str(worm_times.index(my_time)) + '/' + str(len(worm_times)) + '.')
		focal_frame = freeimage.read(working_directory + os.path.sep + a_worm + os.path.sep + my_time + ' bf.png')
		focal_mask = freeimage.read(working_directory + os.path.sep + a_worm + os.path.sep + my_time + ' mask.png').astype('bool')
		a_frame.loc[my_time] = extractFeatures.measure_texture(focal_frame, focal_mask, [my_base.age_texture_codebook, my_base.egg_texture_codebook, my_base.ghost_texture_codebook], [my_base.age_texture_regressor, my_base.egg_texture_regressor, my_base.life_texture_regressor], a_frame.loc[my_time])
	return a_frame

def main():
	return

if __name__ == "__main__":
	main()
