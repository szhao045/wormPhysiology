# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:00:20 2015

@author: Willie
"""

import numpy as np
import pandas as pd
import json
import os
import sklearn.cluster
import sklearn.svm
import pickle
import multiprocessing
import concurrent.futures
import scipy.ndimage

import freeimage
import zplib.image.mask as zplib_image_mask

def color_features(mask_series):
	'''
	Color in some masks with rainbow colors. The colors go in the order red, yellow, green, cyan, blue, magenta.
	'''
	color_array = np.array([
		[255, 0, 0],
		[255, 255, 0],
		[0, 255, 0],
		[0, 255, 255],
		[0, 0, 255],
		[255, 0, 255]		
	]).astype('uint8')
	
	my_shape = list(mask_series[0].shape)
	my_shape.append(3)
	final_picture = np.zeros(my_shape).astype('uint8')
	
	for i in range(0, len(mask_series)):
		final_picture[mask_series[i]] = color_array[i]
	return final_picture

def sample_8star(my_image, my_mask, square_size, sample_number = 1):
	'''
	Given a mask_file and image_file, randomly sample sample_number 8-legged star shapes of size square_size x square_size from within the mask.
	'''
	star_list = []
	my_radius = (square_size - 1)/2	
	my_indices = np.ma.indices(my_image.shape)

	star_mask = np.zeros([square_size, square_size]).astype('bool')
	star_mask[:, my_radius] = True
	star_mask[my_radius, :] = True
	np.fill_diagonal(star_mask, True)
	star_mask = np.fliplr(star_mask)
	np.fill_diagonal(star_mask, True)	

	worm_pixel_number = my_mask[my_mask > 0].shape[0]
	selected_pixels = np.zeros(worm_pixel_number).astype('bool')
	selected_pixels[:sample_number] = True
	selected_pixels = np.random.permutation(selected_pixels)
	selected_pixels = np.array([my_indices[0][my_mask > 0][selected_pixels], my_indices[1][my_mask > 0][selected_pixels]]).transpose()
	for a_pixel in selected_pixels:
		my_square = my_image[a_pixel[0] - my_radius: a_pixel[0] + 1 + my_radius, a_pixel[1] - my_radius: a_pixel[1] + 1 + my_radius]
		if my_square.shape == (square_size, square_size):			
			my_star = my_square[star_mask]		
			star_list.append(my_star)
	return star_list

def train_texture_SVM(training_data_dir, texture_model, samples_per_image, regression_variable, hyperparameters, validated_directory, total_samples = 5000):
	'''
	Train a support vector machine to regress a histogram of texton best matches to the age of a worm.
	'''	
	pickle_file = training_data_dir + os.path.sep + regression_variable.lower() + '_textureSVR.pickle'
	if os.path.isfile(pickle_file):
		with open(pickle_file, 'rb') as my_file:		
			texture_svr = pickle.load(my_file)
		return (texture_svr, hyperparameters)
	
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	training_subdirs = sorted([os.path.join(training_data_dir, subdir) for subdir in os.listdir(training_data_dir) if os.path.isdir(os.path.join(training_data_dir, subdir))])
	histogram_array = []
	age_array = []
	all_data_points = []
	for a_subdir in training_subdirs: 
		total_files = os.listdir(a_subdir)
		data_points = [a_subdir + os.path.sep + a_file for a_file in total_files if a_file.split(' ')[-1] == 'hmask.png']
		all_data_points.extend(list(data_points))
	number_extra_samples = total_samples - len(all_data_points)
	extra_samples = np.random.choice(data_points, size = number_extra_samples, replace = True)
	all_data_points.extend(extra_samples)
	if len(all_data_points) != total_samples:
		raise ValueError('Something is wrong with the size of all_data_points.')
	if len(all_data_points) % my_workers == 0:
		data_point_chunks = np.split(np.array(all_data_points), my_workers)
	else:
		padding = len(all_data_points) % (my_workers - 1)
		data_point_chunks = list(np.split(np.array(all_data_points[:-padding]), my_workers - 1))
		data_point_chunks.append(all_data_points[-padding:])

	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		my_samples = [executor.submit(train_texture_SVM_sample, data_point_chunks[i], regression_variable, samples_per_image, texture_model) for i in range(0, len(data_point_chunks))]
			# Wait for the results and combine them.
		concurrent.futures.wait(my_samples)
		my_samples = [a_job.result() for a_job in my_samples]

	for a_tuple in my_samples:
		histogram_array.extend(a_tuple[0])
		age_array.extend(a_tuple[1])

	if regression_variable + 'Regressor' not in hyperparameters.keys():
		# Do the coarse grid search. Hard-codes including 1/SVM_external_parameters['total_samples'] as a test value for gamma.
		my_workers = min(multiprocessing.cpu_count() - 1, 60)
		coarse_search_parameters = {'C': 2**np.arange(-10, 15).astype('float64'), 'gamma': np.hstack([2**np.arange(-15, 8).astype('float64'), 1/len(age_array)]).astype('float64'), 'epsilon': 2**np.arange(-18, 7).astype('float64')}
		texture_svr = sklearn.svm.SVR()
		my_coarse_search = sklearn.grid_search.GridSearchCV(texture_svr, coarse_search_parameters, n_jobs = my_workers, verbose = 10, cv = 10)
		coarse_answer = my_coarse_search.fit(histogram_array, age_array)
		hyperparameters[regression_variable + 'Regressor'] = coarse_answer.best_params_
		hyperparameters[regression_variable + 'Regressor']['accuracy'] = coarse_answer.best_score_
		with open(validated_directory + os.path.sep + 'hyperparameters.json', 'w') as write_file:
			write_file.write(json.dumps(hyperparameters, sort_keys=True, indent=4))
		
	texture_svr = sklearn.svm.SVR(C = hyperparameters[regression_variable + 'Regressor']['C'], gamma = hyperparameters[regression_variable + 'Regressor']['gamma'], epsilon = hyperparameters[regression_variable + 'Regressor']['epsilon'])
	texture_svr.fit(np.array(histogram_array), np.array(age_array))	

	with open(pickle_file, 'wb') as my_file:		
		pickle.dump(texture_svr, my_file)		
	return (texture_svr, hyperparameters)

def train_texture_SVM_sample(data_points, regression_variable, samples_per_image, texture_model):
	'''
	Get sample histograms for texture classification SVM training.
	'''	
	histogram_array = []
	age_array = []
	
	for a_point in data_points:
		do_point = True		
		
		subdir = os.path.sep.join(a_point.split(os.path.sep)[:-1])
		with open(subdir + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
			my_metadata = json.loads(read_file.read())
		metadata_dict = {my_metadata[i]['timepoint']: my_metadata[i][regression_variable.lower()] for i in range(0, len(my_metadata))}
		egg_age_dict = {my_metadata[i]['timepoint']: my_metadata[i]['egg_age'] for i in range(0, len(my_metadata))}
			
		# Exclude larval animals from ghost_age computation.	
		if regression_variable.lower() == 'ghost_age':
			time_name = a_point.split(os.path.sep)[-1].split(' ')[0]
			if egg_age_dict[time_name] > 0:
				do_point = True
			else:
				do_point = False
				
		if do_point:
			my_time = a_point.split(os.path.sep)[-1].split(' ')[0]
			my_age = metadata_dict[my_time]/24
			mask_file = freeimage.read(a_point)
			image_file = freeimage.read(a_point.replace('hmask.png', 'bf.png'))
			my_samples = sample_8star(image_file, mask_file, 17, sample_number = samples_per_image)
			my_codebook = np.zeros(texture_model.cluster_vectors.shape[0])			
			for a_sample in my_samples:
				my_codebook[texture_model.classify_vector(a_sample)] += 1
			my_codebook = my_codebook/np.sum(my_codebook)
			histogram_array.append(my_codebook)
			age_array.append(my_age)
	return (histogram_array, age_array)

def train_texture(training_data_dir, samples_per_image = 200, regression_variable = 'Age'):
	'''
	Train the model for texture classification.
	'''
	# Use pickled work if I already have it.
	pickle_file = training_data_dir + os.path.sep + regression_variable.lower() + '_textureCodebook.pickle'
	if os.path.isfile(pickle_file):
		with open(pickle_file, 'rb') as my_file:		
			my_texture_model = pickle.load(my_file)
		return my_texture_model

	my_workers = min(multiprocessing.cpu_count() - 1, 60)	
	training_subdirs = sorted([os.path.join(training_data_dir, subdir) for subdir in os.listdir(training_data_dir) if os.path.isdir(os.path.join(training_data_dir, subdir))])
	endings = ['bf.png', 'hmask.png']
	age_ranges = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)]
	cluster_vectors = []
	codebook = []

	samples_dict = {age_range : [] for age_range in age_ranges}	
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		my_samples = [executor.submit(train_texture_samples, subdir, dict(samples_dict), regression_variable, endings, samples_per_image, age_ranges) for subdir in training_subdirs]	
		# Wait for the results and combine them.
		concurrent.futures.wait(my_samples)
		my_samples = [a_job.result() for a_job in my_samples]

	for age_range in samples_dict.keys():
		for some_samples in my_samples:
			samples_dict[age_range].extend(some_samples[age_range])
	for age_range in list(samples_dict.keys()):
		if len(samples_dict[age_range]) == 0:
			samples_dict.pop(age_range)
	
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		my_clusters = [executor.submit(train_texture_clusters, samples_dict[age_range], age_range) for age_range in sorted(list(samples_dict.keys()))]
		
		# Wait for the results and combine them.
		concurrent.futures.wait(my_clusters)
		my_clusters = [a_job.result() for a_job in my_clusters]

	for clustercode in my_clusters:
		codebook.extend(clustercode[1] for clustercode in my_clusters)
		cluster_vectors.extend(clustercode[0].cluster_centers_)
			
	my_texture_model = texture_model(cluster_vectors, codebook)
	with open(pickle_file, 'wb') as my_file:		
		pickle.dump(my_texture_model, my_file)	
	return my_texture_model

def train_texture_clusters(age_samples, age_range):
	'''
	Cluster my samples.
	'''
	my_clusters = sklearn.cluster.MiniBatchKMeans(n_clusters = 60)
	my_clusters.fit(age_samples)
	codebook = [np.mean(age_range) for i in range(0, my_clusters.cluster_centers_.shape[0])]
	return (my_clusters, codebook)

def train_texture_samples(subdir, samples_dict, regression_variable, endings, samples_per_image, age_ranges):
	'''
	Read in samples for texture classification.
	'''
	# Organize some information.
	total_files = os.listdir(subdir)
	data_points = [' '.join(a_file.split('.')[0].split(' ')[0:-1]) for a_file in total_files if a_file.split(' ')[-1] == 'hmask.png']
	with open(subdir + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
		my_metadata = json.loads(read_file.read())
	metadata_dict = {my_metadata[i]['timepoint']: my_metadata[i][regression_variable.lower()] for i in range(0, len(my_metadata))}
	egg_age_dict = {my_metadata[i]['timepoint']: my_metadata[i]['egg_age'] for i in range(0, len(my_metadata))}

	# Exclude larval animals from ghost_age computation.	
	if regression_variable.lower() == 'ghost_age':
		data_points = [a_point for a_point in data_points if egg_age_dict[a_point] > 0]
	
	# Actualyl sample my images.	
	for a_point in data_points:
		my_age = metadata_dict[a_point]/24
		if regression_variable.lower() == 'ghost_age':
			my_age = abs(my_age)
		mask_file = freeimage.read(subdir + os.path.sep + a_point + ' ' + endings[1])
		image_file = freeimage.read(subdir + os.path.sep + a_point + ' ' + endings[0])
		my_samples = sample_8star(image_file, mask_file, 17, sample_number = samples_per_image)
		for age_range in age_ranges:
			if age_range[0] <= my_age < age_range[1]:
				samples_dict[age_range].extend(my_samples)
	return samples_dict

class texture_model():
	'''
	A container class to hold a couple little things I need to classify vectors according to a codebook.
	'''
	def __init__(self, cluster_vectors, codebook):
		'''
		Initialize the class and set some objects as attributes.
		'''
		self.cluster_vectors = np.array(cluster_vectors)
		self.codebook = codebook

	def classify_vector(self, a_vector):
		'''
		Classify a vector according to the codebook.
		'''
		closest_match = np.linalg.norm(np.abs(self.cluster_vectors - a_vector), axis = 1).argmin()
		return closest_match

def measure_texture(my_image, my_mask, texture_models, texture_svr_list, time_series, sample_number = 200):
	'''
	Measure texture characteristics of a worm at the time corresponding to the information given.
	'''
	texture_ages = []
	texture_samples = sample_8star(my_image, my_mask, 17, sample_number = sample_number)
	for i in range(0, len(texture_models)):
		texture_model = texture_models[i]
		my_codebook = np.zeros(texture_model.cluster_vectors.shape[0])			
		for a_sample in texture_samples:
			my_codebook[texture_model.classify_vector(a_sample)] += 1
		my_codebook = my_codebook/np.sum(my_codebook)
		texture_svr = texture_svr_list[i]
		texture_age = texture_svr.predict(my_codebook)[0]
		texture_ages.append(texture_age)
	time_series.loc[['age_texture', 'egg_texture', 'life_texture']] = texture_ages
	time_series.loc['intensity_texture'] = np.mean(my_image[my_mask])
	return time_series

def measure_eggs(egg_mask, focal_mask, time_series):
	'''
	Measure egg characteristics of a worm at the time corresponding to the information given.
	'''
	def characterize_items(labels, region_indices, areas):
		'''
		Measure characteristics of the labeled items from my mask.
		'''	
		item_info = pd.DataFrame(index = region_indices, columns = ['x', 'y', 'Eggs', 'Area'])
		for a_label in region_indices:
			my_look = labels.copy().astype('uint8')
			my_look[labels != a_label] = 0
			my_look[my_look > 0] = -1			
			number_eggs = int(round(areas[a_label-1] / ((1/(0.7**2))*350)))
			item_info.loc[a_label, 'Eggs'] = number_eggs			
			a_worm = np.array(np.where(my_look > 0))
			a_center = np.mean(a_worm, axis = 1)
			item_info.loc[a_label, ['x', 'y']] = a_center			
			item_info.loc[a_label, 'Area'] = areas[a_label-1]			
		return item_info

	(new_labels, new_region_indices, new_areas) = zplib_image_mask.get_areas(egg_mask)
	new_items = characterize_items(new_labels, new_region_indices, new_areas)		
	total_visible_area = new_items.loc[:, 'Area'].sum()	
	total_visible_eggs = new_items.loc[:, 'Eggs'].sum()		
	average_egg_size = new_items[new_items.loc[:, 'Eggs'] == 1].loc[:, 'Area'].mean()
	single_eggs = (new_items.loc[:, 'Eggs'] == 1).sum()
	time_series.loc[['visible_area', 'visible_eggs', 'average_egg_size', 'single_eggs']] = (total_visible_area, total_visible_eggs, average_egg_size, single_eggs)
	
	
	single_egg_mask = np.zeros(new_labels.shape).astype('bool')
	double_egg_mask = np.zeros(new_labels.shape).astype('bool')
	triple_egg_mask = np.zeros(new_labels.shape).astype('bool')
	more_eggs_mask = np.zeros(new_labels.shape).astype('bool')
	for an_item in new_items.index:
		if new_items.loc[an_item, 'Eggs'] == 1:
			single_egg_mask[new_labels == an_item] = True
		if new_items.loc[an_item, 'Eggs'] == 2:
			double_egg_mask[new_labels == an_item] = True
		if new_items.loc[an_item, 'Eggs'] == 3:
			triple_egg_mask[new_labels == an_item] = True
		if new_items.loc[an_item, 'Eggs'] > 3:
			more_eggs_mask[new_labels == an_item] = True	
	colored_areas = color_features([focal_mask, single_egg_mask, double_egg_mask, triple_egg_mask, more_eggs_mask])
	return (time_series, colored_areas)

def measure_autofluorescence(fluorescent_image, worm_mask, time_series):
	'''
	Measure fluorescence characteristics of a worm at the time corresponding to the information given.
	'''
	my_fluorescence = fluorescent_image[worm_mask].copy()
	(intensity_50, intensity_60, intensity_70, intensity_80, intensity_90, intensity_95, intensity_100) = np.percentile(my_fluorescence, np.array([50, 60, 70, 80, 90, 95, 100]).astype('float64'))
	integrated_50 = np.sum(my_fluorescence[my_fluorescence > intensity_50])
	integrated_60 = np.sum(my_fluorescence[my_fluorescence > intensity_60])
	integrated_70 = np.sum(my_fluorescence[my_fluorescence > intensity_70])
	integrated_80 = np.sum(my_fluorescence[my_fluorescence > intensity_80])
	integrated_90 = np.sum(my_fluorescence[my_fluorescence > intensity_90])
	integrated_95 = np.sum(my_fluorescence[my_fluorescence > intensity_95])
	integrated_0 = np.sum(my_fluorescence)
	time_series.loc[['intensity_50', 'intensity_60', 'intensity_70', 'intensity_80', 'intensity_90', 'intensity_95', 'intensity_100', 'integrated_50', 'integrated_60', 'integrated_70', 'integrated_80', 'integrated_90', 'integrated_95', 'integrated_0']] = (intensity_50, intensity_60, intensity_70, intensity_80, intensity_90, intensity_95, intensity_100, integrated_50, integrated_60, integrated_70, integrated_80, integrated_90, integrated_95, integrated_0)
	
	over_0_mask = np.zeros(worm_mask.shape).astype('bool')
	over_50_mask = np.zeros(worm_mask.shape).astype('bool')
	over_60_mask = np.zeros(worm_mask.shape).astype('bool')
	over_70_mask = np.zeros(worm_mask.shape).astype('bool')
	over_80_mask = np.zeros(worm_mask.shape).astype('bool')
	over_90_mask = np.zeros(worm_mask.shape).astype('bool')
	over_0_mask[worm_mask] = True
	over_50_mask[fluorescent_image > intensity_50] = True
	over_50_mask[np.invert(worm_mask)] = False
	over_60_mask[fluorescent_image > intensity_60] = True
	over_60_mask[np.invert(worm_mask)] = False
	over_70_mask[fluorescent_image > intensity_70] = True
	over_70_mask[np.invert(worm_mask)] = False
	over_80_mask[fluorescent_image > intensity_80] = True
	over_80_mask[np.invert(worm_mask)] = False
	over_90_mask[fluorescent_image > intensity_90] = True
	over_90_mask[np.invert(worm_mask)] = False
	colored_areas = color_features([over_0_mask, over_50_mask, over_60_mask, over_70_mask, over_80_mask, over_90_mask])	
	return (time_series, colored_areas)

def measure_size(worm_mask, time_series):
	'''
	Measure size characteristics of a worm at the time corresponding to the information given.
	'''
	total_size = np.sum(worm_mask)
	x_mask = np.max(worm_mask, axis = 1)
	x_range = (x_mask.shape[0] - x_mask[::-1].argmax()) - (x_mask.argmax())
	y_mask = np.max(worm_mask, axis = 0)
	y_range = (y_mask.shape[0] - y_mask[::-1].argmax()) - (y_mask.argmax())
	aspect_ratio = x_range/y_range
	if aspect_ratio < 1:
		aspect_ratio = 1/aspect_ratio
	time_series.loc[['total_size', 'aspect_ratio']] = (total_size, aspect_ratio)
	return time_series

def measure_movementCOM(movement_masks, frame_masks, frames, my_metadata, time_series):
	'''
	Measure movement characteristics of a worm at the time corresponding to the information given.
	'''
	# Calculate the durations over which images are taken.
	movement_times = my_metadata['movement_frame_times']
	unstimulated_duration = movement_times[1]-movement_times[0]
	stimulated_duration_a = movement_times[3]-movement_times[2]
	stimulated_duration_b = movement_times[4]-movement_times[3]	

	# Find centers of mass of masks.
	all_movement = list(movement_masks)
	all_movement.extend(frame_masks)
	centers_of_mass = {}
	the_indices = ['00', '01', '10', '11', '12', 'last_frame', 'now_frame']
	for i in range(0, len(all_movement)):
		a_mask = all_movement[i]
		a_worm = np.array(np.where(a_mask > 0))
		a_center = np.mean(a_worm, axis = 1)
		centers_of_mass[the_indices[i]] = a_center
	
	# Compute the distances moved and movement rates.
	unstimulated_distance = np.linalg.norm(centers_of_mass['00'] - centers_of_mass['01'])
	stimulated_distance_a = np.linalg.norm(centers_of_mass['10'] - centers_of_mass['11'])
	stimulated_distance_b = np.linalg.norm(centers_of_mass['12'] - centers_of_mass['11'])
	bulk_distance = np.linalg.norm(centers_of_mass['now_frame'] - centers_of_mass['last_frame'])	
	unstimulated_rate = unstimulated_distance/unstimulated_duration
	stimulated_rate_a = stimulated_distance_a/stimulated_duration_a
	stimulated_rate_b = stimulated_distance_b/stimulated_duration_b
	fine_movement = np.mean(frames[0][frame_masks[0]] - frames[1][frame_masks[0]])
	time_series.loc[['unstimulated_rate', 'stimulated_rate_a', 'stimulated_rate_b', 'bulk_movement', 'fine_movement']] = (unstimulated_rate, stimulated_rate_a, stimulated_rate_b, bulk_distance, fine_movement)
	
	colored_areas_now = color_features(movement_masks)	
	colored_areas_bulk = color_features(frame_masks)	
	return (time_series, colored_areas_now, colored_areas_bulk)

def measure_great_lawn(bacterial_lawn):
	'''
	Measure the total area and largest diameter of bacterial_lawn in pixels.
	'''
	total_size = np.sum(bacterial_lawn)
	eroded_lawn = scipy.ndimage.morphology.binary_erosion(bacterial_lawn)
	lawn_edge = (~eroded_lawn) & (bacterial_lawn)
	my_indices = np.array(np.ma.nonzero(lawn_edge)).transpose()
	point_a = my_indices[0]
	distances_a = np.sum(np.square(my_indices - point_a), axis = 1)
	point_b = my_indices[distances_a.argmax()]
	distances_b = np.sum(np.square(my_indices - point_b), axis = 1)
	max_diameter = np.sqrt(distances_b.max())
	return (total_size, max_diameter)

def main():
	return

if __name__ == "__main__":
	main()
