# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:31:19 2016

@author: Willie
"""

import os
import pandas as pd
import numpy as np

from zplib.image import colorize as zplib_image_colorize

import basicOperations.imageOperations as imageOperations

def validate_measure(complete_df, directory_bolus, a_measure, a_time, given_worms = []):
	'''
	Pick out a bunch of images and draw boxes around them to validate a_measure at a_time.
	'''
	my_data = rank_worms(complete_df, a_measure, a_time)	
	get_mode = 'worm'
	box_size = 500
	if a_measure in ['cumulative_eggs', 'bulk_movement', 'visible_eggs']:
		get_mode = 'lawn'
		box_size = 800
	elif a_measure in ['intensity_80']:
		get_mode = 'fluorescence'
	
	tercile_colors = [[0, 0, 0], [32767, 32767, 32767], [65535, 65535, 65535]]
	if given_worms == []:
		quintiles = np.array_split(my_data, 5)
		terciles = [quintiles[0], quintiles[2], quintiles[4]] # Make pseudo-terciles from quintiles.
	else:
		terciles = np.array_split(my_data, 3)
		
	# Incorporate the given worms into the workflow properly.	
	if len(given_worms) > 0:
		tercile_worms = [[' '.join(full_worm.split(' ')[:-1]) for full_worm in list(a_tercile.index)] for a_tercile in terciles]	
		tercile_choice_worms = [[a_worm for a_worm in given_worms if a_worm in a_tercile_worms] for a_tercile_worms in tercile_worms]
		tercile_choice_times = [[closest_real_time(complete_df, a_worm, a_time) for a_worm in a_tercile_worms] for a_tercile_worms in tercile_choice_worms]
		tercile_choices = [[tercile_choice_worms[i][j] + ' '  + tercile_choice_times[i][j] for j in range(0, 4)] for i in range(0, 3)]

	# Select new random worms if needed.
	else:
		tercile_choices = [np.random.choice(tercile.index, size = 4, replace = False) for tercile in terciles]
		tercile_choice_worms = [[' '.join(full_time.split(' ')[:-1]) for full_time in a_tercile] for a_tercile in tercile_choices]
		tercile_choice_times = [[full_time.split(' ')[-1] for full_time in a_tercile] for a_tercile in tercile_choices]
	
	# Get the images.
	tercile_images = [[imageOperations.get_worm(directory_bolus, tercile_choice_worms[i][j], tercile_choice_times[i][j], box_size = box_size, get_mode = get_mode) for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]

	# Measure-specific image processing.
	if a_measure in ['life_texture']:
		tercile_images = tercile_images[::-1]
	if a_measure in ['intensity_80']:
		new_tercile_images = []
		for i in range(0, len(tercile_choices)):
			new_tercile_images.append([])
			for j in range(0, len(tercile_choices[i])):
				new_tercile_images[i].append(tercile_images[i][j].astype('uint32')*70)
				new_tercile_images[i][j][new_tercile_images[i][j] > 2**16 - 1] = 2**16 - 1
				new_tercile_images[i][j] = new_tercile_images[i][j].astype('uint16')
		tercile_images = new_tercile_images
	if a_measure in ['bulk_movement']:
		current_time_indices = [[list(complete_df.raw[tercile_choice_worms[i][j]].index).index(tercile_choice_times[i][j]) for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]
		previous_times = [[complete_df.raw[tercile_choice_worms[i][j]].index[current_time_indices[i][j] - 1] for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]
		previous_images = [[imageOperations.get_worm(directory_bolus, tercile_choice_worms[i][j], previous_times[i][j], box_size = box_size, get_mode = get_mode) for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]
		tercile_images = [[np.mean(np.array([previous_images[i][j], tercile_images[i][j]]), axis = 0) for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]
	if a_measure not in ['intensity_80']:	
		images_max = np.max(np.array(tercile_images))
		dtype_max = 2**16-1
		tercile_images = [[(tercile_images[i][j].astype('float64')/images_max*dtype_max).astype('uint16') for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]
	
	# Wrap it in a colored border.
	tercile_images = [[imageOperations.border_box(tercile_images[i][j], border_color = tercile_colors[i], border_width = box_size//15) for j in range(0, len(tercile_choices[i]))] for i in range(0, len(tercile_choices))]
	return (tercile_images, tercile_choices)

def cohort_trace(adult_df, cohort_indices, a_variable, include_ages = True):
	'''
	Return the average cohort trace along with the corresponding ages for the cohort given by cohort_indices for a_variable.
	'''
	cohort_data = adult_df.mloc(adult_df.worms, [a_variable])[cohort_indices, 0, :]
	cohort_data = cohort_data[~np.isnan(cohort_data).all(axis = 1)]
	cohort_data = np.mean(cohort_data, axis = 0)
	age_data = adult_df.ages[:cohort_data.shape[0]]
	if include_ages:
		return (age_data, cohort_data) 
	else:
		return cohort_data

def select_weird_small(complete_df, directory_bolus, automated = True):
	'''
	Try to select the weird, small, long-lived worms and return a list of images for manual validation.
	'''
	# Set up my needed data.
	check_age = '2.0'
	young_adult_data = complete_df.mloc(measures = ['adjusted_size', 'ghost_age'], times = [check_age])[:, :, 0]
	(middle_size, middle_lifespan) = np.median(young_adult_data, axis = 0)
	
	# Group worms according to size and lifespan.
	worm_groups = {}	
	worm_groups['long_big'] = list(np.array(complete_df.worms)[(young_adult_data[:, 0] > middle_size) & (young_adult_data[:, 1] < middle_lifespan)])
	worm_groups['long_small'] = list(np.array(complete_df.worms)[(young_adult_data[:, 0] < middle_size) & (young_adult_data[:, 1] < middle_lifespan)])
	worm_groups['short_big'] = list(np.array(complete_df.worms)[(young_adult_data[:, 0] > middle_size) & (young_adult_data[:, 1] > middle_lifespan)])
	worm_groups['short_small'] = list(np.array(complete_df.worms)[(young_adult_data[:, 0] < middle_size) & (young_adult_data[:, 1] > middle_lifespan)])
	
	# Find closest real timepoint for manual validation/screening.	
	check_worms = []
	if automated:
		check_worms = list(worm_groups['long_small'])
	else:
		check_worms = complete_df.worms
	check_images = [directory_bolus.working_directory + os.path.sep + a_worm + os.path.sep + closest_real_time(complete_df, a_worm, '2.0', egg_mode = True) + ' bf.png' for a_worm in check_worms]
	return (check_images, worm_groups)

def flatten_two_dimensions(three_dimensional_array, axis_to_keep):
	'''
	Flatten two dimensions of a three_dimensional_array, leaving axis_to_keep separate from the other two and returning it as the second dimension.
	'''
	three_dimensional_array = np.swapaxes(three_dimensional_array, axis_to_keep, 2)
	two_dimensional_array = np.reshape(three_dimensional_array, (three_dimensional_array.shape[0]*three_dimensional_array.shape[1], three_dimensional_array.shape[2]))
	return two_dimensional_array

def closest_real_time(complete_df, a_worm, a_time, egg_mode = True):
	'''
	For a_worm at a_time, find the closest real time point.
	'''
	time_split = a_time.split('.')
	hours_time = int(time_split[0])*24 + int(time_split[1])*3
	if not egg_mode:
		time_index = np.nanargmin(np.abs(complete_df.raw[a_worm].loc[:, 'age'] - hours_time))
	else:
		time_index = np.nanargmin(np.abs(complete_df.raw[a_worm].loc[:, 'egg_age'] - hours_time))
	real_time = complete_df.raw[a_worm].index[time_index]
	return real_time

def rank_worms(complete_df, a_variable, a_time, return_all = False, egg_mode = True):
	'''
	Rank worms according to their measured value of a_variable at a_time.
	'''
	if a_time != None:
		my_data = complete_df.mloc(measures = [a_variable], times = [a_time])[:, 0, 0]
		my_index = list(complete_df.worms)
		for i in range(0, len(my_index)):
			a_worm = my_index[i]
			my_time = closest_real_time(complete_df, a_worm, a_time, egg_mode = egg_mode)
			my_index[i] += ' ' + my_time
		my_data = pd.Series(my_data, index = my_index).dropna()
		my_data.sort()
	else:
		my_data = complete_df.mloc(measures = [a_variable])[:, 0, :].copy()
		flat_data = np.ndarray.flatten(my_data)
		true_max = np.nanargmax(flat_data)
		sorted_arguments = np.argsort(flat_data)
		sorted_arguments = sorted_arguments[:np.where(sorted_arguments == true_max)[0] + 1]
		sorted_indices = np.array(np.unravel_index(sorted_arguments, my_data.shape)).transpose()

		the_lowest = [complete_df.worms[sorted_indices[i][0]] + ' ' + closest_real_time(complete_df, complete_df.worms[sorted_indices[i][0]], complete_df.times[sorted_indices[i][1]]) for i in range(0, 20)]
		the_highest = [complete_df.worms[sorted_indices[-i][0]] + ' ' + closest_real_time(complete_df, complete_df.worms[sorted_indices[-i][0]], complete_df.times[sorted_indices[-i][1]]) for i in range(20, 0, -1)]
		together_list = list(the_lowest)
		together_list.extend(the_highest)
		
		together_data = np.concatenate((flat_data[sorted_arguments[:20]], flat_data[sorted_arguments[-20:]]))
		my_data = pd.Series(together_data, index = together_list)
		if return_all:
			the_full = [complete_df.worms[sorted_indices[i][0]] + ' ' + closest_real_time(complete_df, complete_df.worms[sorted_indices[i][0]], complete_df.times[sorted_indices[i][1]]) for i in range(0, len(sorted_indices))]
			return (my_data, the_full)
	return my_data

def get_lifespans(complete_df):
	'''
	Compute lifespans for all worms in complete_df.
	'''
	lifespans = complete_df.mloc(complete_df.worms, ['age', 'ghost_age'], ['0.0'])[:, :, 0]
	lifespans = lifespans[:, 0] - lifespans[:, 1]
	return lifespans

def get_adultspans(complete_df):
	'''
	Compute adultspans for all worms in complete_df.
	'''
	adultspans = complete_df.mloc(complete_df.worms, ['egg_age', 'ghost_age'], ['0.0'])[:, :, 0]
	adultspans = adultspans[:, 0] - adultspans[:, 1]
	return adultspans

def life_cohort_bins(complete_df, my_worms = None, bin_width_days = 2):
	'''
	Compute bins of cohorts of worms.
	'''
	if my_worms == None:
		my_worms = complete_df.worms
	lifespans = get_lifespans(complete_df)/24
	max_lifespan = int(np.ceil(np.max(lifespans)))
	my_bins = [(i*bin_width_days, i*bin_width_days + bin_width_days) for i in range(0, max_lifespan//bin_width_days + 1)]
	life_cohorts = [[] for a_bin in my_bins]
	for i in range(0, lifespans.shape[0]):
		if complete_df.worms[i] in my_worms:
			my_lifespan = lifespans[i]
			my_bin = int(my_lifespan//bin_width_days)
			life_cohorts[my_bin].append(i)
	
	# Remove empty bins and cohorts.
	my_bins = [my_bins[i] for i in range(0, len(life_cohorts)) if len(life_cohorts[i]) > 0]	
	life_cohorts = [a_cohort for a_cohort in life_cohorts if len(a_cohort) > 0]

	# Figure out the average lifespan of each bin.
	bin_lifes = np.zeros(len(my_bins))
	for i in range(0, len(bin_lifes)):
		bin_lifes[i] = np.mean(lifespans[life_cohorts[i]])

	# Color code the bins.
	my_colors = zplib_image_colorize.color_map(bin_lifes/np.max(bin_lifes[~np.isnan(bin_lifes)]))
	my_colors = my_colors/255		
	return (life_cohorts, bin_lifes, my_bins, my_colors)

def adult_cohort_bins(complete_df, my_worms = None, bin_width_days = 2):
	'''
	Compute bins of cohorts of worms.
	'''
	if my_worms == None:
		my_worms = complete_df.worms
	adultspans = get_adultspans(complete_df)/24
	max_adultspans = int(np.ceil(np.max(adultspans)))
	my_bins = [(i*bin_width_days, i*bin_width_days + bin_width_days) for i in range(0, max_adultspans//bin_width_days + 1)]
	life_cohorts = [[] for a_bin in my_bins]
	for i in range(0, adultspans.shape[0]):
		if complete_df.worms[i] in my_worms:
			my_lifespan = adultspans[i]
			my_bin = int(my_lifespan//bin_width_days)
			life_cohorts[my_bin].append(i)
	
	# Remove empty bins and cohorts.
	my_bins = [my_bins[i] for i in range(0, len(life_cohorts)) if len(life_cohorts[i]) > 0]	
	life_cohorts = [a_cohort for a_cohort in life_cohorts if len(a_cohort) > 0]

	# Figure out the average lifespan of each bin.
	bin_lifes = np.zeros(len(my_bins))
	for i in range(0, len(bin_lifes)):
		bin_lifes[i] = np.mean(adultspans[life_cohorts[i]])

	# Color code the bins.
	my_colors = zplib_image_colorize.color_map(bin_lifes/np.max(bin_lifes[~np.isnan(bin_lifes)]))
	my_colors = my_colors/255		
	return (life_cohorts, bin_lifes, my_bins, my_colors)

def main():
	return

if __name__ == "__main__":
	main()
