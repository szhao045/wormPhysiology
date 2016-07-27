# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:47:27 2015

@author: Willie
"""

import numpy as np
import os
import sklearn.cluster
import sklearn.svm
import skimage.feature
import scipy
import scipy.ndimage
import concurrent.futures
import multiprocessing
import shutil 

import freeimage
from zplib.image import mask as zplib_image_mask

def ensure_human(human_dir, work_dir, data_dir):
	'''
	Make sure that the human data directory, human_dir, has everything it needs from the working directory (namely properly corrected bf images and rough background subtraction masks).
	'''
	def test_and_copy(test_file, found_file):
		'''
		Check for test_file file in human_dir, then copy them over from found_file in work_dir if needed.
		'''
		if not os.path.isfile(test_file):
			print('Missing ' + test_file.split(' ')[-1] + ' at ' + test_file + '.')
			if os.path.isfile(found_file):
				print('\tFound corresponding ' + found_file.split(' ')[-1] + ' at ' + found_file + '.')
				print('\tCopying file...')
				shutil.copyfile(found_file, test_file)
				if os.path.isfile(test_file):
					print('\tSuccess!')
				else:
					raise BaseException('\tCOPYING FAILED.')
			else:
				raise BaseException('\tCouldn\'t find corresponding file.')
		return
		
	points_list = []
	for a_subdir in os.listdir(human_dir):
		human_subdir = human_dir + os.path.sep + a_subdir
		for a_file in os.listdir(human_subdir):
			if a_file.split('.')[-1] == 'png':
				file_split = a_file.split(' ')
				points_list.append(a_subdir + os.path.sep + file_split[0])
	points_list = sorted(list(set(points_list)))

	for a_point in points_list:
		# Set up some variables and copy metadata.
		(a_subdir, the_point) = a_point.split(os.path.sep)
		human_subdir = human_dir + os.path.sep + a_subdir
		working_subdir = work_dir + os.path.sep + a_subdir.split(' ')[-1]
		data_subdir = data_dir + os.path.sep + a_subdir.split(' ')[-1]
		base_metadata = human_subdir + os.path.sep + 'position_metadata.json'
		found_metadata = data_subdir + os.path.sep + 'position_metadata.json'
		test_and_copy(base_metadata, found_metadata)

		# Clean up human mask.
		base_test = human_subdir + os.path.sep + the_point
		base_found = working_subdir + os.path.sep + the_point
		test_and_copy(base_test + ' ' + 'hmask.png', base_test + ' ' + 'outline.png')
		print('Cleaning up ' + base_test + ' ' + 'hmask.png' + '.')
		old_mask = freeimage.read(base_test + ' ' + 'hmask.png')
		my_mask = np.zeros(old_mask.shape[:2]).astype('uint8')
		if len(old_mask.shape) == 3:
			my_mask[np.max(old_mask[:, :, :3], axis = 2).astype('float64') > 0] = -1
		elif len(old_mask.shape) == 2:
			my_mask[old_mask > 0] = -1
		else:
			raise ValueError(base_test + ' ' + 'hmask.png' + 'does not have proper dimensions.')
		my_mask[np.invert(zplib_image_mask.get_largest_object(my_mask) > 0)] = 0
		freeimage.write(my_mask, base_test + ' ' + 'hmask.png')

		# Clean up old stuff.
		if os.path.isfile(base_test + ' ' + 'bf.png'):
			os.remove(base_test + ' ' + 'bf.png')
		if os.path.isfile(base_test + ' ' + 'outline.png'):
			os.remove(base_test + ' ' + 'outline.png')

		# Copy over other test masks.
		test_and_copy(base_test + ' ' + 'bf.png', base_found + ' ' + 'bf.png')
		test_and_copy(base_test + ' ' + 'mask.png', base_found + ' ' + 'mask.png')
	return

def string_to_from_dictionary(a_string_or_dictionary):
	'''
	Converts a string or dictionary specifying an SVM's parameters into the alternate form.
	'''
	string_ordering = [
		'image_type', 
		'sampling', 
		'total_samples',
		'square_size',
		'feature_type', 
		'feature_shape' 
	]
	if type(a_string_or_dictionary) == type(''):
		my_string = a_string_or_dictionary
		my_split_string = my_string.split(';')
		my_dictionary = {}
		for i in range(0, len(my_split_string)):		
			if my_split_string[i].isdigit():			
				my_split_string[i] = int(my_split_string[i])
			my_dictionary[string_ordering[i]] = my_split_string[i]
		return my_dictionary
		
	elif type(a_string_or_dictionary) == type({}):
		my_dictionary = a_string_or_dictionary		
		my_string = ';'.join([str(my_dictionary[a_key]).split('.')[0] for a_key in string_ordering])
		return my_string

def select_sample(image_point, vignette_mask, total_sample_number, SVM_external_parameters):
	'''
	Given a mask in a_mask and an image file in image_file, sample points to use for texture classification according to selection_mode and total_sample_number.
	'''	
	def random_select_from_mask(a_mask, sample_number, my_radius):
		'''
		Select sample_number pixels from a_mask which have a clear radius of length my_radius around themselves (i.e. they are not too close to the border of the image).
		'''
		# Ensure that the edges of the image are not included in the mask.
		a_mask[:my_radius, :] = False		
		a_mask[-my_radius:, :] = False
		a_mask[:, :my_radius] = False		
		a_mask[:, -my_radius:] = False

		# Randomly select pixels from the mask.
		pixel_number = a_mask[a_mask > 0].shape[0]
		my_indices = np.ma.indices(my_image.shape)
		selected_pixels = np.zeros(pixel_number).astype('bool')
		selected_pixels[:int(total_sample_number/2)] = True
		selected_pixels = np.random.permutation(selected_pixels)
		selected_pixels = np.array([my_indices[0][a_mask > 0][selected_pixels], my_indices[1][a_mask > 0][selected_pixels]]).transpose()
		return selected_pixels

	# Rescale images and prepare some supporting variables.
	worm_mask = freeimage.read(image_point + ' ' + 'hmask.png').astype('bool')
	my_image = freeimage.read(image_point + ' ' + 'bf.png')
	(my_image, worm_mask, vignette_mask) = scaled_images(my_image, worm_mask, vignette_mask, SVM_external_parameters)
	my_radius = (SVM_external_parameters['square_size'] - 1)/2	
	
	# Select half pixels from the worm itself and half pixels from an area outside the worm but within 50 pixels of the worm.
	selection_mode = SVM_external_parameters['sampling']
	if selection_mode == 'Local_Worm':
		distance_from_worm = scipy.ndimage.morphology.distance_transform_edt(np.invert(worm_mask)).astype('uint16')
		background_mask = np.zeros(worm_mask.shape).astype('bool')	
		background_mask[distance_from_worm > 0] = True
		background_mask[distance_from_worm > worm_mask.shape[0]//20] = False
		background_mask[np.invert(vignette_mask)] = False		
		background_selected_pixels = random_select_from_mask(background_mask, int(total_sample_number/2), my_radius)		
		worm_selected_pixels = random_select_from_mask(worm_mask, int(total_sample_number/2), my_radius)
		
	elif selection_mode == 'Hard_Negative':
		crude_mask = freeimage.read(image_point + ' ' + 'mask.png').astype('bool')
		negatives_mask = crude_mask & np.invert(worm_mask)	
		background_selected_pixels = random_select_from_mask(negatives_mask, int(total_sample_number/2), my_radius)		
		worm_selected_pixels = random_select_from_mask(worm_mask, int(total_sample_number/2), my_radius)

	else:
		raise BaseException('Invalid selection_mode passed from SVM_external_parameters[\'sampling\'].')

	my_pixels = np.vstack([worm_selected_pixels, background_selected_pixels])
	my_squares = [my_image[a_pixel[0] - my_radius: a_pixel[0] + 1 + my_radius, a_pixel[1] - my_radius: a_pixel[1] + 1 + my_radius] for a_pixel in my_pixels]
	my_classifications = np.hstack([np.array([1]*int(total_sample_number/2)), np.array([0]*int(total_sample_number/2))])		
	my_features = feature_computer(my_squares, SVM_external_parameters)
	my_samples = [my_features, my_classifications]
	return my_samples

def scaled_images(my_image, worm_mask, vignette_mask, SVM_external_parameters):
	'''		
	Blur/rescale my images if necessary.
	'''
	def blur_and_shrink_image(an_image, rescale_ratio):
		'''
		Blur images by a Gaussian with a standard deviation of 1, downsample the images by 2x in each dimension, and save out the results to output_paths.
		'''
		rescale_times = int(np.log2(rescale_ratio))
		for i in range(rescale_times):		
			an_image = scipy.ndimage.filters.gaussian_filter(an_image, 1)
			an_image = an_image[::2, ::2]
		return an_image
	
	def rescale_mask(a_mask, rescale_ratio):
		'''
		Rescale a mask by binning and allowing the pixel neighborhood to vote for worm/non-worm.
		'''
		new_shape = (a_mask.shape[0]/rescale_ratio, a_mask.shape[1]/rescale_ratio)
		temporary_shape = (new_shape[0], a_mask.shape[0] // new_shape[0], new_shape[1], a_mask.shape[1] // new_shape[1])
		binned_mask = a_mask.reshape(temporary_shape).mean(-1).mean(1)
		new_mask = np.ones(np.shape(binned_mask)).astype('bool')
		new_mask[binned_mask < 0.5] = False
		return new_mask
	
	if SVM_external_parameters['image_type'] == '2blur_bf':
		vignette_mask = rescale_mask(vignette_mask, 2)
		worm_mask = rescale_mask(worm_mask, 2)
		my_image = blur_and_shrink_image(my_image, 2)
	elif SVM_external_parameters['image_type'] == '4blur_bf':
		vignette_mask = rescale_mask(vignette_mask, 4)
		worm_mask = rescale_mask(worm_mask, 4)
		my_image = blur_and_shrink_image(my_image, 4)
	return (my_image, worm_mask, vignette_mask)

def feature_computer(patch_array, SVM_external_parameters):
	'''
	Computes features for a_patch according to feature_mode, which defaults to 	
	'''
	# Set patch_array according to patch_mode.
	if SVM_external_parameters['feature_shape'] == 'Square':
		patch_array = patch_array
	elif SVM_external_parameters['feature_shape'] == '8-Star':
		# Make a basic 8-star mask.
		square_size = patch_array[0].shape[0]
		my_radius = (square_size - 1)/2	
		star_mask = np.zeros([square_size, square_size]).astype('bool')
		star_mask[:, my_radius] = True
		star_mask[my_radius, :] = True
		np.fill_diagonal(star_mask, True)
		star_mask = np.fliplr(star_mask)
		np.fill_diagonal(star_mask, True)	
		star_mask = np.fliplr(star_mask)
		
		# Make a dummy dimension so that a big star mask can be applied to the big array.
		star_size = np.count_nonzero(star_mask)
		samples_number = patch_array.shape[0]
		my_strides = [0]
		my_strides.extend(star_mask.strides)
		my_strides = tuple(my_strides)		
		big_star_mask = np.ndarray(patch_array.shape, 'bool', star_mask, strides = my_strides)
		patch_array = patch_array[big_star_mask]
		patch_array = np.reshape(patch_array, (samples_number, star_size))
	
	# Compute the features according to feature_mode.
	feature_type = SVM_external_parameters['feature_type'].split('_')[0]
	feature_processing = []
	if len(SVM_external_parameters['feature_type'].split('_')) > 1:
		feature_processing = SVM_external_parameters['feature_type'].split('_')[1:]
	if feature_type == 'Intensity':
		features_vectors = patch_array
	elif feature_type == 'Normed_Intensity':
		features_vectors = patch_array/(2**16)
	elif feature_type == 'HOG':
		features_vectors = [skimage.feature.hog(my_patch, pixels_per_cell=(1, 1), cells_per_block=(1, 1)) for my_patch in patch_array]
		features_vectors = [np.ndarray.flatten(a_feature_vector) for a_feature_vector in features_vectors]

	# Do any additional processing.		
	if 'Normed' in feature_processing:
		features_vectors = features_vectors/(2**16-1)
	return features_vectors


def do_SVM_grid_search(texture_svr_tuple, SVM_external_parameters, search_mode = 'Full'):
	'''
	Do a grid search to find the best hyperparameters to use.
	'''
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	texture_svr = sklearn.svm.NuSVC()

	search_functions = {'Full':sklearn.grid_search.GridSearchCV, 'Fast':sklearn.grid_search.RandomizedSearchCV}

	# Do the coarse grid search. Hard-codes including 1/SVM_external_parameters['total_samples'] as a test value for gamma.
	coarse_search_parameters = {'nu': (np.arange(1, 100)*0.01).astype('float64'), 'gamma': np.hstack([2**np.arange(-15, 4).astype('float64'), 1/SVM_external_parameters['total_samples']]).astype('float64')}
	my_coarse_search = search_functions[search_mode](texture_svr, coarse_search_parameters, n_jobs = my_workers, verbose = 10, cv = 10)
	coarse_answer = my_coarse_search.fit(texture_svr_tuple[1], texture_svr_tuple[2])
	coarse_parameters = coarse_answer.best_params_

	# Do a fine grid search.
	fine_search_parameters = {'nu': (np.arange(1, 200)*(0.01**2) + coarse_parameters['nu'] - (0.01**1)).astype('float64'), 'gamma': np.array([coarse_parameters['gamma']]).astype('float64')}
	my_fine_search = search_functions[search_mode](texture_svr, fine_search_parameters, n_jobs = my_workers, verbose = 10, cv = 10)
	fine_answer = my_fine_search.fit(texture_svr_tuple[1], texture_svr_tuple[2])
	fine_parameters = fine_answer.best_params_
	
	# Do up to 3 even finer grid searches if necessary.
	finer_steps = 0
	while fine_parameters['nu'] == np.min(fine_search_parameters['nu']):
		if finer_steps > 5:
			return fine_parameters
		finer_steps += 1		
		fine_search_parameters = {'nu': (np.arange(1, 200)*(0.01**(2 + finer_steps)) + fine_parameters['nu'] - (0.01**(1+finer_steps))).astype('float64'), 'gamma': np.array([fine_parameters['gamma']]).astype('float64')}
		my_fine_search = search_functions[search_mode](texture_svr, fine_search_parameters, n_jobs = my_workers, verbose = 10, cv = 10)
		fine_answer = my_fine_search.fit(texture_svr_tuple[1], texture_svr_tuple[2])
		fine_parameters = fine_answer.best_params_
	return fine_parameters

def train_texture_SVM(training_image_prefixes, SVM_external_parameters, vignette_mask, hyperparameters = None):
	'''
	Train a support vector machine to classify texture patches as worm or non-worm.
	'''	
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	if hyperparameters == None:
		hyperparameters = {'nu': 0.5, 'gamma': 1/SVM_external_parameters['total_samples']}
	texture_svr = sklearn.svm.NuSVC(nu = hyperparameters['nu'], gamma = hyperparameters['gamma'])		

	feature_array = []
	score_array = []

	total_training_images = len(training_image_prefixes)
	samples_per_image = SVM_external_parameters['total_samples']//total_training_images
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		my_samples = [executor.submit(select_sample, a_point, vignette_mask, SVM_external_parameters = SVM_external_parameters, total_sample_number = samples_per_image) for a_point in training_image_prefixes]
	
	# Wait for the results and combine them.
	concurrent.futures.wait(my_samples)
	my_samples = [a_job.result() for a_job in my_samples]
	for a_tuple in my_samples:
		feature_array.extend(a_tuple[0])
		score_array.extend(a_tuple[1])	
	texture_svr.fit(np.array(feature_array), np.array(score_array))
	return (texture_svr, np.array(feature_array), np.array(score_array))

def classify_pixels(my_pixels, worm_image, selected_pixels, my_mask, texture_svc, SVM_external_parameters):
	'''
	Classify a bunch of pixels.
	'''
	my_feature_vectors = feature_computer(my_pixels, SVM_external_parameters)
	my_answers = texture_svc.predict(my_feature_vectors)
	my_mask[selected_pixels] = my_answers
	return my_mask

def make_neighborhood_view(an_image, square_size):
	'''
	Make a 4-D view of an image that gives me neighborhoods quickly.
	'''
	my_radius = int((square_size - 1)/2)
	padding = [(my_radius, my_radius), (my_radius, my_radius)] + [(0,0) for weird_extras in range(an_image.ndim - 2)]
	padded = np.pad(an_image, padding, mode = 'edge')
	shape = an_image.shape[:2] + (square_size, square_size) + an_image.shape[2:]
	strides = padded.strides[:2]*2 + padded.strides[2:]
	return np.ndarray(shape, padded.dtype, buffer = padded, strides = strides)

def classify_worm_nonworm(worm_image, rough_mask, texture_svc, vignette_mask, SVM_external_parameters, threaded = False, extra_distance = None):
	'''
	Refine a rough worm mask using texture information from the bright field worm image.
	'''
	# Set up some variables we'll need.
	(worm_image, rough_mask, vignette_mask) = scaled_images(worm_image, rough_mask.copy(), vignette_mask, SVM_external_parameters)	
	my_mask = np.zeros(worm_image.shape)
	
	# Select appropriate pixels to classify.
	selected_pixels = np.zeros(worm_image.shape).astype('bool')
	if extra_distance == None:
		extra_distance = rough_mask.shape[0]//20
	if extra_distance > 0:
		close_to_rough = selected_pixels.copy()	
		distance_from_rough = scipy.ndimage.morphology.distance_transform_edt(np.invert(rough_mask)).astype('uint16')
		close_to_rough = np.zeros(rough_mask.shape).astype('bool')	
		close_to_rough[distance_from_rough > 0] = True
		close_to_rough[distance_from_rough > extra_distance] = False
		selected_pixels[close_to_rough]	= True
	selected_pixels[rough_mask]	= True
	selected_pixels[np.invert(vignette_mask)] = False
	neighborhood_view = make_neighborhood_view(worm_image, SVM_external_parameters['square_size'])
	my_pixels = neighborhood_view[selected_pixels]
		
	if threaded:
		# Split up the pixels of the image into nice chunks.		
		print('\tParallelizing classification now...', flush = True)
		my_workers = min(multiprocessing.cpu_count() - 1, 60)
		chunk_size = int(np.ceil(selected_pixels.shape[0]/my_workers))
		pixel_chunks = [my_pixels[x:x + chunk_size] for x in range(0, my_pixels.shape[0], chunk_size)]

		# Classify the pixels in a threaded way.
		print('\tSubmitting classification now...', flush = True)
		with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
			chunk_masks = [executor.submit(my_pixels, worm_image, selected_pixels, my_mask.copy(), texture_svc, SVM_external_parameters) for pixel_chunk in pixel_chunks]
	
		# Wait for the results and combine them.
		concurrent.futures.wait(chunk_masks)
		print('\tCombining classification now...', flush = True)
		chunk_masks = [a_job.result() for a_job in chunk_masks]
		for a_chunk in chunk_masks:
			my_mask[a_chunk > 0] = a_chunk[a_chunk > 0]	
	else:
		# Do the classification using only one thread.
		my_mask = classify_pixels(my_pixels, worm_image, selected_pixels, my_mask.copy(), texture_svc, SVM_external_parameters)
	return my_mask.astype('bool')

def main():
	return

if __name__ == "__main__":
	main()
