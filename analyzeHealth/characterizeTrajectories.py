# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 06:02:01 2015

@author: Willie
"""

import sys
import pickle
import os
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import sklearn.cluster
import json
import multiprocessing
import concurrent.futures

import basicOperations.folderStuff as folderStuff
import measurePhysiology.organizeData as organizeData
import analyzeHealth.computeStatistics as computeStatistics

class CompleteWormDF():
	'''
	A class analogous to the complete_df I used for the Framingham Heart Study, containing several useful methods for analyzing worms.
	'''
	def __init__(self, directory_bolus, save_directory, extra_arguments = {}):
		'''
		Initialize the kind of dataframe that I really need! No more pandas slowness and nonsense.
		'''		
		# Set some default values if parameters not given in extra_arguments.
		if 'copy_health' not in extra_arguments.keys():
			extra_arguments['copy_health'] = False
		if 'copy_directory' not in extra_arguments.keys():
			extra_arguments['copy_directory'] = None
		if 'adult_only' not in extra_arguments.keys():
			extra_arguments['adult_only'] = False
		if 'use_old' not in extra_arguments.keys():
			if sys.platform == 'win32':
				extra_arguments['use_old'] = True
			elif sys.platform == 'linux':
				extra_arguments['use_old'] = False
		if 'smooth_mode' not in extra_arguments.keys():
			extra_arguments['smooth_mode'] = 'one'
		if 'renew_health' not in extra_arguments.keys():		
			extra_arguments['renew_health'] = False
		if 'key_measures' not in extra_arguments.keys():		
			extra_arguments['key_measures'] = ['intensity_80', 'cumulative_eggs', 'bulk_movement', 'adjusted_size', 'life_texture']
			
		# Read in the raw data and set up some basic information.		
		self.save_directory = save_directory
		self.key_measures = extra_arguments['key_measures']
		
		self.adult_only = extra_arguments['adult_only']
		data_directories = directory_bolus.data_directories[:directory_bolus.done]
		if os.path.isdir(data_directories[0]) and not extra_arguments['use_old']:
			self.raw = self.read_trajectories(data_directories, save_directory)
		else:
			self.raw = self.read_trajectories(directory_bolus.windows_health, save_directory)
		if extra_arguments['copy_health']:
			if extra_arguments['copy_directory'] == None:
				extra_arguments['copy_directory'] = directory_bolus.windows_health
			folderStuff.ensure_folder(extra_arguments['copy_directory'])
			for a_worm in self.raw.keys():
				self.raw[a_worm].to_csv(extra_arguments['copy_directory'] + os.path.sep + a_worm + '.tsv', sep = '\t')
		self.extra_changed = False
		
		if not extra_arguments['renew_health']:
			self.load_extra(directory_bolus)
		else:
			self.extra_data = {}

		self.worms = sorted(list(self.raw.keys()))
		self.measures = list(self.raw[self.worms[0]].columns)
		all_shapes = []		
		for a_worm in self.raw.keys():
			my_shape = self.raw[a_worm].shape
			all_shapes.append(my_shape)
		my_shape = np.max(np.array(all_shapes), axis = 0)
		max_times = my_shape[0]
		max_measurements = my_shape[1]

		# Set up ways to look up raw indices of string names of variables and worms.
		self.worm_indices = {self.worms[i]:i for i in range(0, len(self.worms))}
		self.measure_indices = {self.measures[i]:i for i in range(0, len(self.measures))}
		self.times = [str(a_time//8) + '.' + str(a_time-8*(a_time//8)) for a_time in list(range(0, max_times))]
		self.time_indices = {self.times[i]:i for i in range(0, len(self.times))}
		self.ages = [int(a_time.split('.')[0]) + int(a_time.split('.')[1])/8 for a_time in self.times]
		
		# Fill in my frame with actual data!
		self.data = np.empty((len(self.worms), max_measurements, max_times))
		self.data.fill(np.nan)
		for i in range(0, len(self.worms)):
			a_worm = self.worms[i]
			worm_data = self.raw[a_worm].values
			(time_shape, measure_shape) = worm_data.shape
			self.data[i, 0:measure_shape, 0:time_shape] = worm_data.transpose()

		# Set up some information for re-scaling.
		self.means = np.empty(len(self.measures[:-3]))
		self.means[:] = np.nan
		self.stds = np.empty(len(self.measures[:-3]))
		self.stds[:] = np.nan

		# Process the egg counts, smooth trajectories.
		self.process_eggs()
		if 'total_size' in self.measures:
			self.adjust_machine_bias(directory_bolus)
			self.smooth_trajectories(directory_bolus, extra_arguments)	
			
		# Do some quick re-scaling.
		self.time_normalized_data()		
		self.scale_normalized_data()

		# Add rates of change.
		if 'total_size' in self.measures:
			self.add_rates()
			self.scale_normalized_data()

		# Add SVM-health data.
		if 'total_size' in self.measures:
			self.add_healths()

		# Save out my extra data if it has changed.
		if self.extra_changed:
			self.save_extra(directory_bolus)
			
	def load_extra(self, directory_bolus):
		'''		
		Load my extra data if I can find it.		
		'''
		frame_string = ''
		if self.adult_only:
			frame_string += 'adult_'
		extra_file = directory_bolus.windows_health + os.path.sep + frame_string + 'extra_data.pickle'
		extra_backup = directory_bolus.human_directory + os.path.sep + frame_string + 'extra_data.pickle'
		if os.path.isfile(extra_file):
			with open(extra_file, 'rb') as my_file:		
				self.extra_data = pickle.load(my_file)
		elif os.path.isfile(extra_backup):
			with open(extra_backup, 'rb') as my_file:		
				self.extra_data = pickle.load(my_file)
		else:
			self.extra_data = {}
		return 
		
	def save_extra(self, directory_bolus):
		'''
		Save my extra data if it's changed.
		'''
		frame_string = ''
		if self.adult_only:
			frame_string += 'adult_'
		extra_file = directory_bolus.windows_health + os.path.sep + frame_string + 'extra_data.pickle'
		extra_backup = directory_bolus.human_directory + os.path.sep + frame_string + 'extra_data.pickle'
		if os.path.isdir(directory_bolus.windows_health):
			with open(extra_file, 'wb') as my_file:		
				pickle.dump(self.extra_data, my_file)	
		if os.path.isdir(directory_bolus.working_directory):
			with open(extra_backup, 'wb') as my_file:		
				pickle.dump(self.extra_data, my_file)	
		return
				
	def mloc(self, worms = None, measures = None, times = None):
		'''
		Selects information from my dictionary of normalized time dataframes as if it were a 3-dimensional array.
		'''
		mloc_data = self.data
		if worms != None:
			worm_indices = [self.worm_indices[a_worm] for a_worm in worms]
			mloc_data = mloc_data[worm_indices, :, :]
		if measures != None:
			measure_indices = [self.measure_indices[a_measure] for a_measure in measures]
			mloc_data = mloc_data[:, measure_indices, :]
		if times != None:
			times = [self.time_indices[a_time] for a_time in times]
			mloc_data = mloc_data[:, :, times]
		return mloc_data

	def flat_data(self, measures):
		'''
		Send back a list of flattened arrays.
		'''
		my_flats = [np.ndarray.flatten(self.mloc(measures = [a_measure])) for a_measure in measures]
		return my_flats

	def get_center_std(self, base_dictionary):
		'''
		Get the centers and standard deviations of all variables in base_dictionary.
		'''
		overall_frame = []
		for a_worm in base_dictionary.keys():
			my_frame = base_dictionary[a_worm].values
			my_frame = my_frame[~np.isnan(my_frame).any(axis = 1)]
			overall_frame.append(my_frame)
		overall_frame = np.vstack(overall_frame)		
		my_means = np.mean(overall_frame, axis = 0)
		my_stds = np.std(overall_frame, axis = 0)
		my_means[-4:] = 0
		my_stds[-4:] = 1
		return (my_means, my_stds)

	def outcome_correlation(self, a_measurement, time_range = list(range(0, 1000)), worm_range = None, outcome = 'age'):
		'''
		See if various attributes are correlated with age.
		'''
		if worm_range == None:
			worm_range = self.worms
		my_information = self.mloc(worm_range, [a_measurement], time_range).values.flatten()
		my_outcomes = self.mloc(worm_range, [outcome], time_range).values.flatten()
		(my_r, my_p) = scipy.stats.stats.pearsonr(my_information, my_outcomes)
		return (my_r, my_r**2, my_p)
	
	def identify_worms(self, a_measurement, a_time):
		'''
		For a_measurement at a_time, list the worms from lowest to highest values.
		'''
		my_ordering = self.mloc(measures = [a_measurement], times = [a_time])[:, 0, 0]
		my_ranks = my_ordering.argsort()
		my_ordering.sort()
		my_worms = []
		for i in range(0, my_ordering.shape[0]):
			if ~np.isnan(my_ordering[i]):
				my_worms.append(self.worms[my_ranks[i]])
		return my_worms
	
	def scale_normalized_data(self):
		'''
		Rescale normalized data so that it's in terms of standard deviations from the overall mean.
		'''
		for var_index in range(len(self.measures[:-3])):
			if np.isnan(self.means[var_index]):
				a_var = self.measures[var_index]			
				my_data = np.ndarray.flatten(self.mloc(measures = [a_var]))
				my_data = my_data[~np.isnan(my_data)]
				self.means[var_index] = np.mean(my_data)
				self.stds[var_index] = np.std(my_data)
				self.data[:, var_index, :] = (self.data[:, var_index, :] - self.means[var_index])/self.stds[var_index]		
		return
	
	def time_normalized_data(self):
		'''
		Normalize my data so that times between time points are equal.
		'''
		normed_data = []
		max_times = 0
		for a_worm in self.worms:
			worm_index = self.worm_indices[a_worm]
			birth_time = self.mloc([a_worm], ['age'])
			if self.adult_only:
				birth_time = self.mloc([a_worm], ['egg_age'])
			death_time = self.mloc([a_worm], ['ghost_age'])
			start_index = np.abs(birth_time[~np.isnan(birth_time)]).argmin()
			end_index = np.abs(death_time[~np.isnan(death_time)]).argmin()

			age_array = self.mloc([a_worm], ['age'])[0, 0, start_index: end_index + 1]
			adult_start = age_array[0]
			
			normed_array = np.empty((len(self.measures), 1000)).astype('float')
			normed_array.fill(np.nan)
		
			adult_span = age_array[-1] - age_array[0]
			time_points = int(adult_span//3) + 1
			max_times = max(max_times, time_points)

			for i in range(0, time_points):
				i_age = adult_start + (i*3)
	
				age_sort = np.sort(np.append(age_array, i_age))
				my_indices = np.where(age_sort == i_age)[0]
				my_indices = np.where(abs(age_sort - i_age) < 0.001)[0]
				indices_shape = my_indices.shape[0]
	
				if indices_shape == 2:
					normed_array[:, i] = self.data[worm_index, :, start_index + my_indices[0]]
				elif indices_shape == 1:
					before_position = my_indices[0] - 1
					after_position = my_indices[0]
					before_age = age_array[before_position]
					after_age = age_array[after_position]
					age_range = after_age - before_age
					before_portion = (after_age - i_age)/age_range
					after_portion = (i_age - before_age)/age_range
					normed_array[:, i] = before_portion*self.data[worm_index, :, start_index + before_position] + after_portion*self.data[worm_index, :, start_index + after_position]
				else:
					raise BaseException('Too many instances of i_age found in age_sort.')
			normed_data.append(normed_array)
		normed_data = np.array(normed_data)[:, :, :max_times]
		self.data = normed_data
		self.times = [str(a_time//8) + '.' + str(a_time-8*(a_time//8)) for a_time in list(range(0, max_times))]
		self.time_indices = {self.times[i]:i for i in range(0, len(self.times))}
		self.ages = [int(a_time.split('.')[0]) + int(a_time.split('.')[1])/8 for a_time in self.times]
		return
					
	def add_column(self, column_data, column_index, column_name):
		'''
		Add a new column of processed data to my array.
		'''
		self.data = np.concatenate([self.data[:, :column_index, :], column_data, self.data[:, column_index:, :]], axis = 1)	
		self.measures.insert(column_index, column_name)
		self.measure_indices = {self.measures[i]:i for i in range(0, len(self.measures))}
		
		mean_std_index = column_index
		if column_index < 0:
			mean_std_index = len(self.measures) + column_index
		self.means = np.concatenate([self.means[:mean_std_index], [np.nan], self.means[mean_std_index:]], axis = 0)	
		self.stds = np.concatenate([self.stds[:mean_std_index], [np.nan], self.stds[mean_std_index:]], axis = 0)	
		return
		
	def process_eggs(self):
		'''
		Add a "cumulative eggs" variable to each worm dataframe in worm_frames.
		'''
		if 'visible_eggs' in self.measures and 'visible_area' in self.measures:
			new_data_eggs = []
			new_data_area = []
			for i in range(0, len(self.worms)):
				# Extract some needed information.
				a_worm = self.worms[i]
				visible_eggs = self.mloc([a_worm], ['visible_eggs'])
				visible_area = self.mloc([a_worm], ['visible_area'])
				egg_started = self.mloc([a_worm], ['egg_age'])
				start_index = np.abs(egg_started[~np.isnan(egg_started)]).argmin()
				
				# Eliminate false positive eggs from count.
				base_eggs = visible_eggs[0, 0, start_index - 1]	
				cumulative_eggs = np.array(visible_eggs)
				cumulative_eggs = cumulative_eggs - base_eggs
				cumulative_eggs[:, :, :start_index] = 0
				base_area = visible_area[0, 0, start_index - 1]	
				cumulative_area = np.array(visible_area)
				cumulative_area = cumulative_area - base_area
				cumulative_area[:, :, :start_index] = 0
		
				# Find running maximum of visible eggs to use as cumulative eggs laid.
				cumulative_eggs = np.maximum.accumulate(cumulative_eggs, axis = 2)			
				cumulative_area = np.maximum.accumulate(cumulative_area, axis = 2)			
				new_data_eggs.append(cumulative_eggs)			
				new_data_area.append(cumulative_area)			
	
			# Add my eggs into the columns.
			new_data_eggs = np.concatenate(new_data_eggs, axis = 0)
			column_index = self.measure_indices['visible_eggs']
			self.add_column(new_data_eggs, column_index, 'cumulative_eggs')
			new_data_area = np.concatenate(new_data_area, axis = 0)
			column_index = self.measure_indices['visible_area']
			self.add_column(new_data_area, column_index, 'cumulative_area')
		return
						
	def smooth_trajectory(self, a_variable, extra_arguments):
		'''
		Smooth trajectories for a single variable.
		'''
		# Prepare some overall variables.
		my_index = self.measure_indices[a_variable]
		unsmoothed_data = self.data[:, my_index, :]
		smoothed_data = np.zeros(unsmoothed_data.shape)
		
		# Forward and back-fill data in preparation for smoothing.
		for j in range(0, self.data.shape[0]):
			my_data = unsmoothed_data[j, :]
			all_ind = np.where(~np.isnan(my_data))
			if len(all_ind[0]) > 0:
				ind = all_ind[0]
				first, last = ind[0], ind[-1]
				my_data[:first] = my_data[first]
				my_data[last + 1:] = my_data[last]
			# Do the actual smoothing.
			for i in range(0, extra_arguments['smooth_iterations']):
				# Use a Savitzky-Golay filter.
				if extra_arguments['smooth_mode'] == 'savitzky_golay':
					my_data = scipy.signal.savgol_filter(my_data, window_length = extra_arguments['windowLength'], polyorder = extra_arguments['polyorder'])						
				# Use a running mean.
				elif extra_arguments['smooth_mode'] == 'running_mean':
					cumulative_data = np.cumsum(np.insert(my_data, 0, 0)) 
					running_mean = (cumulative_data[extra_arguments['windowLength']:] - cumulative_data[:-extra_arguments['windowLength']]) / extra_arguments['windowLength'] 
					for i in range(0, (extra_arguments['windowLength'] - 1) // 2):
						running_mean = np.insert(running_mean, 0, running_mean[0])
						running_mean = np.insert(running_mean, -1, running_mean[-1])
					my_data = running_mean
				# Use a median filter.
				elif extra_arguments['smooth_mode'] == 'median':
					window_radius = (extra_arguments['windowLength'] - 1) // 2
					view_array = np.zeros((len(my_data), extra_arguments['windowLength']), dtype = my_data.dtype)
					view_array[:, window_radius] = my_data
					for i in range(0, window_radius):
						j = window_radius - i
						view_array[j:, i] = my_data[:-j]
						view_array[:j, i] = my_data[0]
						view_array[:-j, -(i+1)] = my_data[j:]
						view_array[-j:, -(i+1)] = my_data[-1]
					my_data = np.median(view_array, axis=1)
			smoothed_data[j, :] = my_data
		# Return the smoothed data!
		return smoothed_data
			
	def smooth_trajectories(self, directory_bolus, extra_arguments):
		'''
		Apply a Savitzsky-Golay filter to smooth my data.
		'''	
		# Don't do anything if smooth_mode is set to None.
		if extra_arguments['smooth_mode'] == None:			
			return

		# Load in my parameters.
		if os.path.exists(directory_bolus.windows_health + os.path.sep + 'smooth_parameters.json'):
			with open(directory_bolus.windows_health + os.path.sep + 'smooth_parameters.json', 'r') as read_file:
				smooth_settings = json.loads(read_file.read())
		elif os.path.exists(directory_bolus.human_directory + os.path.sep + 'smooth_parameters.json'):
			with open(directory_bolus.human_directory + os.path.sep + 'smooth_parameters.json', 'r') as read_file:
				smooth_settings = json.loads(read_file.read())
		else:
			unsmoothed_df = CompleteWormDF(directory_bolus, self.save_directory, {'smooth_mode': None})
			(smooth_settings, results_dict) = grid_search_smooth(unsmoothed_df, directory_bolus)
			
		# Save my parameters in both locations if they are only in one.			
		if os.path.exists(directory_bolus.human_directory):
			if not os.path.exists(directory_bolus.human_directory + os.path.sep + 'smooth_parameters.json'):
				with open(directory_bolus.human_directory + os.path.sep + 'smooth_parameters.json', 'w') as write_file:
					write_file.write(json.dumps(smooth_settings, sort_keys=True, indent=4))
		if os.path.exists(directory_bolus.windows_health):
			if not os.path.exists(directory_bolus.windows_health + os.path.sep + 'smooth_parameters.json'):
				with open(directory_bolus.windows_health + os.path.sep + 'smooth_parameters.json', 'w') as write_file:
					write_file.write(json.dumps(smooth_settings, sort_keys=True, indent=4))

		# Provide some feedback on progression of smoothing as you do it.
		for a_variable in sorted(list(smooth_settings.keys())):
			print('\tSmoothing ' + a_variable + '.')
			variable_index = self.measure_indices[a_variable]	
			if extra_arguments['smooth_mode'] == 'one':
				smoothed_data = self.smooth_trajectory(a_variable, {'smooth_iterations': 3, 'polyorder': 1, 'windowLength': 9, 'smooth_mode': 'savitzky_golay'})
			else:				
				smoothed_data = self.smooth_trajectory(a_variable, {'smooth_iterations': smooth_settings[a_variable]['iterations'], 'polyorder': smooth_settings[a_variable]['order_polynomial'], 'windowLength': smooth_settings[a_variable]['window_len'], 'smooth_mode': smooth_settings[a_variable]['method']})
			self.data[:, variable_index, :] = smoothed_data
		return

	def read_trajectories(self, data_directories, save_directory):
		'''
		Reads in trajectories from .tsv files in working_directory's measured_health subdirectory and then groups them together nicely in a WormData class. Also adds meta-information from metadata in data_directory.
		'''
		if len(data_directories) == 1:
			data_directories = data_directories[0]
		
		# List of worms to exclude.
		not_yet_done = [
#			'2016.03.31 spe-9 16 005',
#			'2016.03.31 spe-9 16 132',
#			'2016.03.31 spe-9 16 196',
#			'2016.03.31 spe-9 16 200',
#			'2016.03.31 spe-9 16 201',
#			'2016.03.31 spe-9 16 202',
#			'2016.03.31 spe-9 16 204',
#			'2016.03.31 spe-9 16 208'
		]
		never_eggs = [
			'2016.02.26 spe-9 11D 130',
			'2016.02.26 spe-9 11C 085',
			'2016.02.29 spe-9 12A 07',
			'2016.03.25 spe-9 15B 003',
			'2016.03.25 spe-9 15B 126',
			'2016.03.25 spe-9 15A 43',
			'2016.03.04 spe-9 13C 67',
			'2016.03.31 spe-9 16 154',
			'2016.05.02 spe-9 age-1 osxi346 17C 08',
			'2016.05.02 spe-9 age-1 osxi346 17C 09',
		]
		
		# Read in my measured_health data from the scope.
		if type(data_directories) == type([]):
			health_directories = [data_directory + os.path.sep + 'measured_health' for data_directory in data_directories]
			my_tsvs = []
			for health_directory in health_directories:
				my_tsvs.extend([health_directory + os.path.sep + a_file for a_file in os.listdir(health_directory) if a_file.split('.')[-1] == 'tsv'])

			# Exclude worms.
			for a_worm in not_yet_done:
				if a_worm + '.tsv' in my_tsvs:
					print('\tSkipping ' + a_worm + ', it is not yet done processing.')
					my_tsvs.remove(a_worm + '.tsv')
			for a_worm in never_eggs:
				worm_file = [a_dir for a_dir in health_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir][0] + os.path.sep + a_worm.split(' ')[-1] + '.tsv'
				if worm_file in my_tsvs:
					print('\tSkipping ' + a_worm + ', it never laid eggs.')
					my_tsvs.remove(worm_file)		

			worm_frames = {a_file.split(os.path.sep)[-3].replace(' Run ', ' ') + ' ' + a_file.split(os.path.sep)[-1].split('.')[-2]: pd.read_csv(a_file, sep = '\t', index_col = 0) for a_file in my_tsvs}		

		# Read in my measured_health data from my special directory.
		elif type(data_directories) == type(''):
			my_tsvs = [a_file for a_file in os.listdir(data_directories) if a_file.split('.')[-1] == 'tsv']
			
			# Exclude worms.
			for a_worm in not_yet_done:
				if a_worm + '.tsv' in my_tsvs:
					print('\tSkipping ' + a_worm + ', it is not yet done processing.')
					my_tsvs.remove(a_worm + '.tsv')
			for a_worm in never_eggs:
				if a_worm + '.tsv' in my_tsvs:
					print('\tSkipping ' + a_worm + ', it never laid eggs.')
					my_tsvs.remove(a_worm + '.tsv')		
			
			# Actually read them in.
			worm_frames = {a_file[:-4]: pd.read_csv(data_directories + os.path.sep + a_file, sep = '\t', index_col = 0) for a_file in my_tsvs}
		return worm_frames
	
	def adjust_machine_bias(self, directory_bolus, size_df = None, renew_regression = False):
		'''
		Adjust total_size for any bias in measurement by the automated method as compared to the manually drawn masks.
		'''	
		if renew_regression:
			if size_df is None:
				size_df = organizeData.validate_generated_masks(directory_bolus.working_directory, directory_bolus.human_directory)
			machine_measures = size_df.loc[:, 'F_Size'].values.astype('float64')
			human_measures = size_df.loc[:, 'H_Size'].values.astype('float64')
			(m, b) = np.polyfit(machine_measures, human_measures, 1)
		else:
			(m, b) = (0.69133050735497115, 8501.9616379946274)
		
		old_size = self.mloc(measures = ['total_size'])
		old_size_index = self.measure_indices['total_size']
		new_size = (m*old_size) + b
		self.add_column(new_size, old_size_index, 'adjusted_size')
		return

	def add_rates(self, rate_variables = None):
		'''
		Add rates of change for some variables.
		'''
		if rate_variables == None:
			rate_variables = ['cumulative_area', 'cumulative_eggs', 'adjusted_size']
		for a_variable in rate_variables:
			print('\tComputing rate of change for ' + a_variable + '.')				
			
			# Compute rates of change.
			my_data = self.mloc(measures = [a_variable])
			my_rate = my_data[:, :, 1:] - my_data[:, :, :-1]
			zfiller = np.zeros(my_data.shape[:2])
			my_rate = np.concatenate([my_rate, zfiller[:, :, np.newaxis]], axis = 2)

			# Actually add the rates data.
			old_index = self.measure_indices[a_variable]
			self.add_column(my_rate, old_index, a_variable + '_rate')
		return

	def add_healths(self):
		'''
		Add 'health' measurements to both extra_data and to my columns.
		'''
		# These are the original measures from which my health measurements are composed.
		health_measures = {
			'autofluorescence': ['intensity_80'],		
			'size': ['adjusted_size', 'adjusted_size_rate'],		
			'eggs': ['cumulative_eggs', 'cumulative_eggs_rate'],		
			'texture': ['life_texture'],		
			'movement': ['bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'],
		}
		
		# Actually do the SVM and regression if it's not in my loaded data. Otherwise just add it to the data.
		all_physiology = []
		specific_healths = sorted(list(health_measures.keys()))
		for a_health in specific_healths:
			print('\tAdding health measure: ' + a_health + '.', flush = True)
			if a_health not in self.extra_data.keys():
				print('\t\tComputing health measure: ' + a_health + '.', flush = True)
				self.extra_changed = True
				(variable_data, svr_data, life_data) = computeStatistics.svr_data(self, health_measures[a_health], dependent_variable = 'ghost_age')
				column_data = np.expand_dims(svr_data, axis = 1)
				self.extra_data[a_health] = column_data
				all_physiology.extend(health_measures[a_health])
			self.add_column(self.extra_data[a_health], -3, a_health)		
		self.scale_normalized_data()

		# Now do it for the overall health measure.
		if 'health' not in self.extra_data.keys():
			print('\t\tComputing overall health measure.', flush = True)
			self.extra_changed = True
			(variable_data, svr_data, life_data) = computeStatistics.svr_data(self, all_physiology, dependent_variable = 'ghost_age')
			column_data = np.expand_dims(svr_data, axis = 1)
			self.extra_data['health'] = column_data
		self.add_column(self.extra_data['health'], -3, 'health')
		self.scale_normalized_data()
		return
	
	def display_names(self, my_var):
		'''
		Just get the display name for my_var.
		'''
		fancy_names = {
			'intensity_90': 'Autofluorescence 90th Percentile Intensity', 
			'intensity_80': 'Autofluorescence 80th Percentile Intensity', 
			'cumulative_eggs': 'Cumulative Oocytes Laid',
			'cumulative_eggs_rate': 'Oocyte Laying Rate',
			'visible_eggs': 'Visible Laid Oocytes',
			'cumulative_area': 'Cumulative Area of Eggs Laid',
			'total_size': 'Cross-Sectional Size', 
			'age_texture': 'Textural Age',
			'bulk_movement': 'Movement',
			'stimulated_rate_a': 'Movement (Stimulated A)',
			'stimulated_rate_b': 'Movement (Stimulated B)',
			'unstimulated_rate': 'Movement Rate (Unstimulated)',
			'area': 'Area',
			'life_texture': 'Textural Degradation',
			'adjusted_size': 'Cross-Sectional Size',
			'adjusted_size_rate': 'Size Rate of Change',
			'great_lawn_area': 'Bacterial Lawn Area', 
			'texture': 'Texture Prognosis',
			'eggs': 'Reproductive Prognosis',
			'autofluorescence': 'Autofluorescence Prognosis',
			'movement': 'Movement Prognosis',
			'size': 'Body Size Prognosis',
			'health': 'Overall Prognosis'
		}
		if my_var in fancy_names.keys():
			fancy_name = fancy_names[my_var]
		else:
			fancy_name = my_var
		return fancy_name
	
	def display_variables(self, an_array, my_var):
		'''
		Convert an_array of my_var data to display units.
		'''
		my_units = {
			'intensity_90': 'Standard Deviations', 
			'intensity_80': 'Standard Deviations', 
			'cumulative_eggs': 'Cumulative Oocytes Laid',
			'cumulative_eggs_rate': 'Oocytes Laid Per Hours',
			'cumulative_area': 'mm^2',
			'visible_eggs': 'Number of Oocytes',
			'total_size': 'mm^2', 
			'age_texture': 'Textural Age', 
			'bulk_movement': 'mm Displacement Per Hours',
			'stimulated_rate_a': 'mm/s',
			'stimulated_rate_b': 'mm/s',
			'unstimulated_rate': 'mm/s',
			'area': 'mm^2',
			'life_texture': 'Texture Prediction (Days Remaining)',
			'adjusted_size': 'mm^2',
			'adjusted_size_rate': 'mm^2 Per Hour',
			'great_lawn_area': 'mm^2', 
			'texture': 'Predicted Days of Life Remaining',
			'eggs': 'Predicted Days of Life Remaining',
			'autofluorescence': 'Predicted Days of Life Remaining',
			'movement': 'Predicted Days of Life Remaining',
			'size': 'Predicted Days of Life Remaining',
			'health': 'Predicted Days of Life Remaining'
		}
		unit_multipliers = {
			'intensity_90': None, 
			'intensity_80': None, 
			'cumulative_eggs': 1,
			'cumulative_eggs_rate': 1/3,
			'cumulative_area': (1.304/1000)**2,
			'visible_eggs': 1,
			'total_size': (1.304/1000)**2, 
			'age_texture': 1, 
			'bulk_movement': (1.304/1000)/3,
			'stimulated_rate_a': (1.304/1000),
			'stimulated_rate_b': (1.304/1000),
			'unstimulated_rate': (1.304/1000),
			'area': 1/100000,
			'life_texture': -1,
			'adjusted_size': (1.304/1000)**2,
			'adjusted_size_rate': ((1.304/1000)**2)/3,
			'great_lawn_area': (1.304/1000)**2, 
			'texture': (-1/24),
			'eggs': (-1/24),
			'autofluorescence': (-1/24),
			'movement': (-1/24),
			'size': (-1/24),
			'health': (-1/24),
		}

		if my_var in my_units.keys():
			my_unit = my_units[my_var]
			fancy_name = self.display_names(my_var)
			unit_multiplier = unit_multipliers[my_var]
		else:
			my_unit = 'Raw Numbers'
			fancy_name = 'No Fancy Name for ' + my_var
			unit_multiplier = 1
	
		if my_unit != 'Standard Deviations' and unit_multiplier != None:
			my_index = self.measures.index(my_var)
			my_mean = self.means[my_index]
			my_std = self.stds[my_index]
			new_array = unit_multiplier*((an_array*my_std) + my_mean)
		else:
			new_array = an_array
		return (new_array, my_unit, fancy_name)

def grid_search_variable(unsmoothed_df, a_variable, life_array, total_tests, measure_count, savgol_parameters, running_mean_parameters, median_parameters):
	'''
	Grid search smoothing parameters for a_variable.
	'''
	i = 0
	variable_index = str(unsmoothed_df.measure_indices[a_variable] + 1)
	my_results = pd.DataFrame(index = list(range(total_tests)), columns = ['r**2', 'iterations', 'window_len', 'order_polynomial', 'method'])
	
	# Test the Savitzky-Golay method.
	for a_window_len in savgol_parameters['window_len']:
		for an_order_polynomial in savgol_parameters['order_polynomial']:
			if an_order_polynomial < a_window_len:
				for an_iterations in savgol_parameters['iterations']:
					i += 1
					print('Up to ' + a_variable + ' (' + variable_index + '/' + measure_count + ') ' + str(i) + '/' + str(total_tests) + '.')
					smoothed_data = unsmoothed_df.smooth_trajectory(a_variable, {'windowLength': a_window_len, 'polyorder': an_order_polynomial, 'smooth_iterations': an_iterations, 'smooth_mode': 'savitzky_golay'})
					smoothed_data = np.ndarray.flatten(smoothed_data)
					my_r = computeStatistics.quick_pearson(smoothed_data, life_array)
					my_results.loc[i, 'r**2'] = my_r
					my_results.loc[i, 'iterations'] = an_iterations
					my_results.loc[i, 'window_len'] = a_window_len
					my_results.loc[i, 'order_polynomial'] = an_order_polynomial						
					my_results.loc[i, 'method'] = 'savitzky_golay'						

	# Test the running mean method.
	for a_window_len in running_mean_parameters['window_len']:
		for an_iterations in running_mean_parameters['iterations']:
			i += 1
			print('Up to ' + a_variable + ' (' + variable_index + '/' + measure_count + ') ' + str(i) + '/' + str(total_tests) + '.')
			smoothed_data = unsmoothed_df.smooth_trajectory(a_variable, {'windowLength': a_window_len, 'smooth_iterations': an_iterations, 'smooth_mode': 'running_mean'})
			smoothed_data = np.ndarray.flatten(smoothed_data)
			my_r = computeStatistics.quick_pearson(smoothed_data, life_array)
			my_results.loc[i, 'r**2'] = my_r
			my_results.loc[i, 'iterations'] = an_iterations
			my_results.loc[i, 'window_len'] = a_window_len
			my_results.loc[i, 'order_polynomial'] = 0
			my_results.loc[i, 'method'] = 'running_mean'						

	# Test the median method.
	for a_window_len in median_parameters['window_len']:
		for an_iterations in median_parameters['iterations']:
			i += 1
			print('Up to ' + a_variable + ' (' + variable_index + '/' + measure_count + ') ' + str(i) + '/' + str(total_tests) + '.')
			smoothed_data = unsmoothed_df.smooth_trajectory(a_variable, {'windowLength': a_window_len, 'smooth_iterations': an_iterations, 'smooth_mode': 'median'})
			smoothed_data = np.ndarray.flatten(smoothed_data)
			my_r = computeStatistics.quick_pearson(smoothed_data, life_array)
			my_results.loc[i, 'r**2'] = my_r
			my_results.loc[i, 'iterations'] = an_iterations
			my_results.loc[i, 'window_len'] = a_window_len
			my_results.loc[i, 'order_polynomial'] = 0
			my_results.loc[i, 'method'] = 'median'						
	return my_results

def grid_search_smooth(unsmoothed_df, directory_bolus):
	'''
	For each variable in self.measures, grid-search the best parameters to use for smoothing it.
	'''
	# These are the variable values to test.
	smooth_modes = ['running_mean', 'median', 'savitzky_golay']
	savgol_parameters = {
		'iterations': [1, 2, 3],
		'window_len': [3, 5, 7, 9],
		'order_polynomial': [1, 2, 3, 4, 5]
	}
	running_mean_parameters = {
		'iterations': [1, 2, 3],
		'window_len': [3, 5, 7, 9]
		}
	median_parameters = {
		'iterations': [1, 2, 3],
		'window_len': [3, 5, 7, 9]
		}
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	
	# Count up how many conditions we're really testing.
	total_tests = 0
	for an_iterations in savgol_parameters['iterations']:
		for a_window_len in savgol_parameters['window_len']:
			for an_order_polynomial in savgol_parameters['order_polynomial']:
				if an_order_polynomial < a_window_len:
					total_tests += 1
	for an_iterations in median_parameters['iterations']:
		for a_window_len in median_parameters['window_len']:
			total_tests += 1
	for an_iterations in running_mean_parameters['iterations']:
		for a_window_len in running_mean_parameters['window_len']:
			total_tests += 1		
	
	# Actually compute r^2 values.
	results_dict = {}
	life_array = np.ndarray.flatten(unsmoothed_df.mloc(measures = ['ghost_age'])[:, 0, :])
	measure_variables = unsmoothed_df.measures[:-3]
	measure_count = str(len(measure_variables))
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		smooth_parameters = [executor.submit(grid_search_variable, unsmoothed_df, a_variable, life_array, total_tests, measure_count, savgol_parameters, running_mean_parameters, median_parameters) for a_variable in measure_variables]
	concurrent.futures.wait(smooth_parameters)
	smooth_parameters = [a_job.result() for a_job in smooth_parameters]
	results_dict = {a_variable: smooth_parameters[unsmoothed_df.measure_indices[a_variable]] for a_variable in measure_variables}
	
	# Process our results a bit to select the best parameters and then save them out.	
	top_results = {a_variable: dict(results_dict[a_variable].loc[results_dict[a_variable]['r**2'].idxmax(), :]) for a_variable in results_dict.keys()}
	if os.path.isdir(directory_bolus.windows_health):
		with open(directory_bolus.windows_health + os.path.sep + 'smooth_parameters.json', 'w') as write_file:
			write_file.write(json.dumps(top_results, sort_keys=True, indent=4))
	if os.path.isdir(directory_bolus.human_directory):
		with open(directory_bolus.human_directory + os.path.sep + 'smooth_parameters.json', 'w') as write_file:
			write_file.write(json.dumps(top_results, sort_keys=True, indent=4))
	return (top_results, results_dict)

def read_pincus(general_directory_bolus, save_dir = r'C:\Users\Willie\Desktop\save_dir', my_folder = r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2011 Pincus PLoS Genetics', adult_only = False):
	'''
	Read in Zach Pincus's data as a dataframe and return it.
	'''
	time_frame = pd.read_csv(my_folder + os.path.sep + 'timecourse.csv')
	summary_frame = pd.read_csv(my_folder + os.path.sep + 'summary.csv')
	summary_frame.index = summary_frame.loc[:, 'worm']
	my_worms = [a_worm for a_worm in list(set(list(summary_frame.index))) if '2010-06-28 mir-239' not in a_worm]
	my_tsvs = [a_file for a_file in os.listdir(my_folder + os.path.sep + 'measured_health') if a_file.split('.')[-1] == 'tsv']
	if len(my_tsvs) < len(my_worms):
		print('Generating worm dfs.')
		for an_index in time_frame.index:
			my_worm = time_frame.loc[an_index, 'worm']
			my_lifespan = summary_frame.loc[my_worm, 'lifespan']
			time_frame.loc[an_index, 'lifespan'] = my_lifespan
		time_frame.loc[:, 'lifespan'] = time_frame.loc[:, 'lifespan']*24
		time_frame.loc[:, 'age'] = time_frame.loc[:, 'age']*24
		if adult_only:
			time_frame.loc[:, 'age'] = time_frame.loc[:, 'age'] - 75
		time_frame.loc[:, 'ghost_age'] = time_frame.loc[:, 'age'] - time_frame.loc[:, 'lifespan']
		column_list = list(time_frame.columns)
		column_list.remove('age')
		column_list.remove('fluor_pc_0')
		column_list.append('age')
		time_frame = time_frame[column_list]
		time_frame.loc[:, 'egg_age'] = 0
		
		my_worms = list(summary_frame.loc[:, 'worm'])
		for i in range(0, len(my_worms)):
			a_worm = my_worms[i]
			print('\t' + str(i+1) + '/' + str(len(my_worms)) + ', ' + a_worm)
			a_frame = time_frame[time_frame.loc[:, 'worm'] == a_worm] 
			col_list = list(a_frame.columns)
			col_list.remove('worm')
			col_list.remove('timepoint')
			a_frame = a_frame[col_list]
			extra_fill = a_frame.iloc[-1]
			extra_fill.loc['age'] = extra_fill.loc['lifespan']		
			extra_fill.loc['ghost_age'] = 0
			a_frame = pd.concat([a_frame, pd.DataFrame(extra_fill).transpose()])
			extra_fill = a_frame.iloc[0]
			extra_fill.loc['age'] = 0		
			extra_fill.loc['ghost_age'] = -1*extra_fill.loc['lifespan']
			a_frame = pd.concat([pd.DataFrame(extra_fill).transpose(), a_frame])
			a_frame.to_csv(my_folder + os.path.sep + 'measured_health' + os.path.sep + a_worm.replace(':', ',') + '.tsv', sep = '\t')
	
	directory_bolus = folderStuff.DirectoryBolus(general_directory_bolus.working_directory, general_directory_bolus.human_directory, data_directories = [my_folder + os.path.sep + 'measured_health'], extra_directories = [None], experiment_directories = [None], annotation_directories = [None], ready = 1)
#	raise BaseException('')	
	
	complete_df = CompleteWormDF(directory_bolus, save_dir, {'key_measures': ['texture_remaining_age', 'motion_remaining_age', 'area', 'tritc_total_95th'], 'use_old': False})
	complete_df.extra_data['summary'] = summary_frame

	def check_worms(pincus_df):
		'''
		Check for worms that have all data.
		'''
		full_worms = []
		for a_worm in pincus_df.worms:
			worm_data = pincus_df.mloc(worms = [a_worm], measures = pincus_df.key_measures)[0, :, :]
			has_measures = ~np.isnan(worm_data).all(axis = 1)
			if all(has_measures):
				print(a_worm, has_measures)
				full_worms.append(a_worm)
		return full_worms
		
	complete_df.extra_data['full_worms'] = check_worms(complete_df)
	return complete_df

def separate_variation_sources(trajectory_a, trajectory_b):
	'''
	Quantify how much differences in rate, starting point, and direction affect the overall difference between trajectory_a and trajectory_b.
	'''	
	def difference_between(trajectory_a, trajectory_b):
		'''
		Quantify the overall difference between trajectory_a and trajectory_b.
		'''
		difference_frame = trajectory_a - trajectory_b
		mean_difference = np.mean([np.linalg.norm(difference_frame[an_index, :]) for an_index in range(0, difference_frame.shape[0])])
		return mean_difference
	
	def normalize_start(trajectory_a, trajectory_b):
		'''
		Align the starting locations of trajectory_a and trajectory_b to that of trajectory_a. 
		'''
		starting_distance = trajectory_a[0, :] - trajectory_b[0, :]
		trajectory_b = trajectory_b + starting_distance
		return (trajectory_a, trajectory_b)
	
	def normalize_distance(trajectory_a, trajectory_b):
		'''
		Equalize the rates of change of trajectory_a and trajectory_b to that of trajectory_a. 
		'''
		a_speed = np.mean(np.linalg.norm(trajectory_a[0:-1, :] - trajectory_a[1:, :], axis = 1))
		b_differences = trajectory_b[1:, :] - trajectory_b[0:-1, :] 
		b_speed = np.mean(np.linalg.norm(b_differences, axis = 1))
		speed_ratio = a_speed/b_speed
		new_trajectory_b = np.zeros(trajectory_b.shape)
		new_trajectory_b += trajectory_b[0]
		for i in range(0, b_differences.shape[0]):		
			new_trajectory_b[i+1:] += speed_ratio*b_differences[i]
		return (trajectory_a, new_trajectory_b)
	
	def normalize_displacement(trajectory_a, trajectory_b):
		'''
		Equalize the rates of change of trajectory_a and trajectory_b to that of trajectory_a. 
		'''
		a_speed = np.linalg.norm(trajectory_a[0, :] - trajectory_a[-1, :])
		b_speed = np.linalg.norm(trajectory_b[0, :] - trajectory_b[-1, :])
		b_differences = trajectory_b[1:, :] - trajectory_b[0:-1, :] 
		speed_ratio = a_speed/b_speed
		new_trajectory_b = np.zeros(trajectory_b.shape)
		new_trajectory_b += trajectory_b[0]
		for i in range(0, b_differences.shape[0]):		
			new_trajectory_b[i+1:] += speed_ratio*b_differences[i]
		return (trajectory_a, new_trajectory_b)
	
	def vector_project(my_vector, a_direction):
		'''
		Computes the vector projection of my_vector on to a_direction.
		'''
		unit_direction = a_direction/np.linalg.norm(a_direction)
		component_length = np.linalg.norm(a_direction)
		if component_length > 0:
			magnitude_result = np.array(my_vector).dot(a_direction)/component_length
		else:
			magnitude_result = 0
		final_result = magnitude_result*unit_direction
		return final_result
	
	def rotate_2D(a_point, an_angle):
		'''
		Rotate a_point by an_angle.
		'''
		rotation_matrix = np.array([[np.cos(an_angle), -np.sin(an_angle)], [np.sin(an_angle), np.cos(an_angle)]])
		rotated_point = np.dot(rotation_matrix, a_point)
		return rotated_point
	
	def back_to_reality(a_in_plane, leftover, spanning1, spanning2):
		'''
		Go back to the full N-dimensional regular space representation of a point.
		'''
		reality_point = np.zeros(spanning1.shape)
		reality_point = reality_point + a_in_plane[0]*spanning1
		reality_point = reality_point + a_in_plane[1]*spanning2
		reality_point = reality_point + leftover
		return reality_point
	
	def plane_of_basis(a_point, spanning1, spanning2):
		'''
		Project a_point into the plane formed by spanning1 and spanning2, and return its coordinates in that basis.
		'''
		component1 = vector_project(a_point, spanning1)
		leftover = a_point - component1	
		component1 = np.linalg.norm(component1)
		component2 = vector_project(a_point, spanning2)
		leftover = leftover - component2	
		component2 = np.linalg.norm(component2)
		a_in_plane = np.array([component1, component2])	
		return (a_in_plane, leftover)
	
	def normalize_direction(trajectory_a, trajectory_b):
		'''
		Rotate trajectory_b so that its overall direction is the same as that of trajectory_a.
		'''
		# Find orthonormal basis vectors that span the plane defined by the two trajectories.
		a_start = trajectory_a[0, :]
		b_start = trajectory_b[0, :]
		trajectory_a = trajectory_a - a_start
		trajectory_b = trajectory_b - b_start
		a_direction = trajectory_a[-1, :] - trajectory_a[0, :]
		b_direction = trajectory_b[-1, :] - trajectory_b[0, :]
		b_without_a = b_direction - vector_project(b_direction, a_direction)
		if np.linalg.norm(b_without_a) < 10**(-10):
			# Do something in the unusual case that the directions of trajectory_a and trajectory_b are co-linear.
			if np.dot(a_direction, b_direction) < 0:
				trajectory_a = trajectory_a + a_start
				trajectory_b = -1*trajectory_b + b_start
				return (trajectory_a, trajectory_b)
			else:
				trajectory_a = trajectory_a + a_start
				trajectory_b = trajectory_b + b_start
				return (trajectory_a, trajectory_b)			
		spanning1 = a_direction/np.linalg.norm(a_direction)
		spanning2 = b_without_a/np.linalg.norm(b_without_a)
		
		# Project your two trajectories into the orthonormal space, keeping track of components that are orthogonal to it.
		(b_in_plane, leftover_trajectory) = plane_of_basis(b_direction, spanning1, spanning2)
		b_plane_angle = np.arctan2(b_in_plane[1], b_in_plane[0])
	
		leftovers = []
		b_in_plane = []
		for i in range(0, trajectory_b.shape[0]):
			(plane_point, leftover_vector) = plane_of_basis(trajectory_b[i, :], spanning1, spanning2)
			leftovers.append(leftover_vector)
			b_in_plane.append(plane_point)
		
		# Rotate the trajectories inside the 2-dimensional space so that the angles match up.
		rotated_b = []	
		for i in range(0, len(b_in_plane)):
			a_point = b_in_plane[i]
			rotated_point = rotate_2D(a_point, -1*b_plane_angle)
			rotated_b.append(rotated_point)
		
		# Return to regular space.	
		trajectory_a = trajectory_a + a_start
		normalized_b = []
		for i in range(0, len(rotated_b)):
			normalized_point = back_to_reality(rotated_b[i], leftovers[i], spanning1, spanning2)
			normalized_b.append(normalized_point)
		normalized_b = normalized_b + b_start
		return (trajectory_a, normalized_b)	
	
	trajectory_b_list = []
	trajectory_a = np.array(trajectory_a)
	trajectory_b = np.array(trajectory_b)
	
	total_difference = difference_between(trajectory_a, trajectory_b)
	trajectory_b_list.append(trajectory_b.copy())	
	
	(trajectory_a, trajectory_b) = normalize_start(trajectory_a, trajectory_b)
	nostart_difference = difference_between(trajectory_a, trajectory_b)
	trajectory_b_list.append(trajectory_b.copy())	
	
	(trajectory_a, trajectory_b) = normalize_displacement(trajectory_a, trajectory_b)
	nostartnorate_difference = difference_between(trajectory_a, trajectory_b)
	trajectory_b_list.append(trajectory_b.copy())	

	(trajectory_a, trajectory_b) = normalize_direction(trajectory_a, trajectory_b)
	nostartnoratenoangle_difference = difference_between(trajectory_a, trajectory_b)
	trajectory_b_list.append(trajectory_b.copy())	
	
	start_difference = (total_difference - nostart_difference)/total_difference 
	rate_difference = (nostart_difference - nostartnorate_difference)/total_difference 
	direction_difference = (nostartnorate_difference - nostartnoratenoangle_difference)/total_difference 
	path_difference = (nostartnoratenoangle_difference)/total_difference 
	variation_dictionary = {'Direction': direction_difference, 'Rate': rate_difference, 'Path': path_difference, 'Start': start_difference}	
	return (variation_dictionary, trajectory_a, trajectory_b_list)

def cohort_variation(complete_df, trajectory_PCA, my_variables = None):
	'''
	Figure out how the paths differ between different lifespan cohorts.
	'''
	(life_cohorts, bin_lifes, my_bins, my_colors) = computeStatistics.life_cohort_bins(complete_df, my_worms = complete_df.worms, bin_width_days = 2)
	
	# Plot the actual stuff.
	if my_variables == None:
		my_variables = trajectory_PCA.measures
	cohort_PC_trajectories = []
	cohort_lifes = []	
	for i in range(0, len(life_cohorts)):
		if len(life_cohorts[i]) > 0:
			a_cohort = life_cohorts[i]
			cohort_data = complete_df.mloc(complete_df.worms, my_variables)[a_cohort, :, :]
			cohort_data = np.mean(cohort_data, axis = 0)
			cohort_data = cohort_data.transpose()
			cohort_data = cohort_data[~np.isnan(cohort_data).any(axis = 1)]
			cohort_PC_trajectory = computeStatistics.project_PCA(cohort_data, trajectory_PCA)
			cohort_PC_trajectories.append(cohort_PC_trajectory)
			cohort_lifes.append(bin_lifes[i])
	
	cohort_lifes = [('%0.2f' % a_life).zfill(5) for a_life in cohort_lifes]
	start_frame = pd.DataFrame(columns = cohort_lifes, index = cohort_lifes)
	path_frame = pd.DataFrame(columns = cohort_lifes, index = cohort_lifes)
	direction_frame = pd.DataFrame(columns = cohort_lifes, index = cohort_lifes)
	rate_frame = pd.DataFrame(columns = cohort_lifes, index = cohort_lifes)
	for i in range(0, len(cohort_PC_trajectories)):
		for j in range(0, len(cohort_PC_trajectories)):
			trajectory_a = cohort_PC_trajectories[i]
			trajectory_b = cohort_PC_trajectories[j]
			max_length = np.max([trajectory_a.shape[0], trajectory_b.shape[0]])
			trajectory_a = resample(trajectory_a, max_length)			
			trajectory_b = resample(trajectory_b, max_length)		
			(variation_dictionary, trajectory_a, trajectory_b_list) = separate_variation_sources(trajectory_a, trajectory_b)
			start_frame.iloc[i, j] = variation_dictionary['Start']
			path_frame.iloc[i, j] = variation_dictionary['Path']
			direction_frame.iloc[i, j] = variation_dictionary['Direction']
			rate_frame.iloc[i, j] = variation_dictionary['Rate']
	my_strs = ['start', 'path', 'direction', 'rate']
	for i in range(0, len((start_frame, path_frame, direction_frame, rate_frame))):
		a_frame = (start_frame, path_frame, direction_frame, rate_frame)[i]
		a_frame = a_frame.replace(np.inf, np.nan)
		a_frame = a_frame.replace(-np.inf, np.nan)
		my_data = np.ndarray.flatten(a_frame.values).astype('float32')
		my_data = my_data[~np.isnan(my_data)]
		print(my_strs[i], 'mean', np.mean(my_data), 'median', np.median(my_data))

	return (start_frame, path_frame, direction_frame, rate_frame)

def resample(a_trajectory, new_points):
	'''
	Resample a_trajectory to have new_points points.
	'''
	a_trajectory = a_trajectory[~np.isnan(a_trajectory).any(axis = 1)]
	(my_points, my_dimensions) = a_trajectory.shape
	new_trajectory = np.zeros((new_points, my_dimensions))
	for i in range(0, my_dimensions):
		new_trajectory[:, i] = np.interp(np.arange(new_points)/new_points, np.arange(my_points)/my_points, a_trajectory[:, i])
	return new_trajectory


def main():
	return

if __name__ == "__main__":
	main()
