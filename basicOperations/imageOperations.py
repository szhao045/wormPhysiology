# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:15:34 2015

@author: Willie
"""

import os
import numpy as np
import json
import random
import shutil
import scipy.ndimage
import pandas as pd
import pickle

import freeimage
from zplib.image import mask as zplib_image_mask

import basicOperations.folderStuff as folderStuff

def renormalize_together(images_list):
	'''
	For images from images_list, renormalize them with the same scaling.
	'''
	images_max = np.max(np.array(images_list))
	images_dtype = images_list[0].dtype
	dtype_max = np.iinfo(images_dtype).max
	images_list = [(images_list[i].astype('float64')/images_max*dtype_max).astype(images_dtype) for i in range(0, len(images_list))]
	images_list = [border_box(images_list[i], border_color = (-1, -1, -1), border_width = 1) for i in range(0, len(images_list))]
	return images_list

def white_worms(my_images, my_masks, box_size = None):
	'''
	Cut out images of worms with white backgrounds.
	'''
	my_squares = []
	masked_squares = []
	for i in range(0, len(my_images)):
		# Select the  images we're working with.		
		my_image = my_images[i]
		my_mask = my_masks[i]		
		
		# Cut out the box we want.
		my_image[np.invert(my_mask)] = 0
		(square_mask, actual_size) = bound_box(my_mask, box_size = box_size)
		my_square = my_image[square_mask]	
		my_square = np.reshape(my_square, actual_size)
		box_radii = (box_size, box_size)
	
		# Do some checks for the box size to keep it a square..	
		if actual_size[0] < box_radii[0]:
			my_square = np.concatenate([my_square, np.zeros((box_radii[0] - actual_size[0], my_square.shape[1])).astype('uint16')], axis = 0)
		if actual_size[1] < box_radii[1]:
			my_square = np.concatenate([my_square, np.zeros((my_square.shape[0], box_radii[1] - actual_size[1])).astype('uint16')], axis = 1)		
		if (my_square.shape[0]//2, my_square.shape[1]//2) != box_radii:
			raise BaseException('')
		my_squares.append(my_square)
		
		# Mask the mask.		
		masked_square = np.invert(my_mask[square_mask])	
		masked_square = np.reshape(masked_square, actual_size)
		masked_squares.append(masked_square)
	
	# Make the background white now.
	my_squares = renormalize_together(my_squares)		
	for i in range(0, len(my_squares)):
		a_worm = my_squares[i]
		masked_square = masked_squares[i]
		a_worm[masked_square] = -1
	return my_squares
	
def get_worm(directory_bolus, a_worm, a_time, box_size = None, get_mode = 'worm'):
	'''
	Get a cut-out of a_worm at a_time.
	'''
	# Cut out the chunk that I want from the right image.	
	image_file = directory_bolus.working_directory + os.path.sep + a_worm + os.path.sep + a_time + ' ' + 'bf.png'
	if get_mode == 'fluorescence':
		worm_directory = [a_dir for a_dir in directory_bolus.data_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir][0] + os.path.sep + a_worm.split(' ')[-1]
		image_file = worm_directory + os.path.sep + a_time + ' ' + 'green_yellow_excitation_autofluorescence.png'
		my_image = corrected_fluorescence(image_file)
	else:
		my_image = freeimage.read(image_file)
	
	# Get the right mask.
	if get_mode in ['worm', 'fluorescence']:
		mask_file = directory_bolus.working_directory + os.path.sep + a_worm + os.path.sep + a_time + ' ' + 'mask.png'
	elif get_mode == 'lawn':
		worm_directory = [a_dir for a_dir in directory_bolus.data_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir][0] + os.path.sep + a_worm.split(' ')[-1]
		mask_file = worm_directory + os.path.sep + 'great_lawn.png'
	if os.path.exists(mask_file):	
		my_mask = freeimage.read(mask_file).astype('bool')
	else:
		raise BaseException('Can\'t access files.')

	# Mask out non-worm for fluorescence mode.
	if get_mode == 'fluorescence':
		my_image[np.invert(my_mask)] = 0

	# Ensure that the final square is the right shape.
	(square_mask, actual_size) = bound_box(my_mask, box_size = box_size)
	my_square = my_image[square_mask]	
	my_square = np.reshape(my_square, actual_size)
	box_size = (box_size, box_size)
	if actual_size[0] < box_size[0]:
		my_square = np.concatenate([my_square, np.zeros((box_size[0] - actual_size[0], my_square.shape[1])).astype('uint16')], axis = 0)
	if actual_size[1] < box_size[1]:
		my_square = np.concatenate([my_square, np.zeros((my_square.shape[0], box_size[1] - actual_size[1])).astype('uint16')], axis = 1)		
	if (my_square.shape[0]//2, my_square.shape[1]//2) != box_size:
		raise BaseException('')
	return my_square

def corrected_fluorescence(image_location):
	'''
	Read in a fluroescence image corrected for flatfield and hot pixels.
	'''
	# Figure out where everything is.
	split_pathsep = image_location.split(os.path.sep)	
	experiment_directory = os.path.sep.join(split_pathsep[:-2])
	calibration_directory = experiment_directory + os.path.sep + 'calibrations'
	time_point = split_pathsep[-1].split(' ')[0]
	hot_threshold = 500	
	pickle_file = experiment_directory + os.path.sep + 'super_vignette.pickle'
	if os.path.isfile(pickle_file):
		with open(pickle_file, 'rb') as my_file:		
			super_vignette = pickle.load(my_file)
	else:
		raise BaseException('Missing super vignette!')

	# Read in image and apply vignetting.
	image_path = image_location
	raw_image = freeimage.read(image_path)
	raw_image[np.invert(super_vignette)] = 0

	# Correct for flatfield.
	flatfield_path = calibration_directory + os.path.sep + time_point + ' ' + 'fl_flatfield.tiff'
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


def border_box(a_box, border_color, border_width = 15):
	'''
	Wrap a_box, which is a numpy array, in a border of border_width pixels and of border_color color.
	'''
	# Set up a colored array of the right shape.
	if len(a_box.shape) == 2:
		(my_width, my_height) = a_box.shape
		color_array = np.empty((my_width + 2*border_width, my_height + 2*border_width, 3), dtype = a_box.dtype)
		color_array[border_width:-border_width, border_width:-border_width, 0] = a_box.copy()
		color_array[border_width:-border_width, border_width:-border_width, 1] = a_box.copy()
		color_array[border_width:-border_width, border_width:-border_width, 2] = a_box.copy()
	elif len(a_box.shape) == 3:
		(my_width, my_height, my_depth) = a_box.shape		
		color_array = np.empty((my_width + 2*border_width, my_height + 2*border_width, 3), dtype = a_box.dtype)
		color_array[border_width:-border_width, border_width:-border_width, :] = a_box.copy()
		
	# Color in the border.
	color_array[-border_width:, :, :] = border_color
	color_array[:border_width, :, :] = border_color
	color_array[:, -border_width:, :] = border_color
	color_array[:, :border_width, :] = border_color
	return color_array

def bound_box(a_mask, box_size = None):
	'''
	Defines a square box bounding a_mask. The size of the box defaults to the minimum size needed to fit a_mask inside. If a_mask is too close to the edge to center properly in box_size, 
	'''
	# Figure out the edges of the mask.
	x_mask = np.max(a_mask, axis = 1)
	x_range = (x_mask.argmax(), x_mask.shape[0] - x_mask[::-1].argmax())
	y_mask = np.max(a_mask, axis = 0)
	y_range = (y_mask.argmax(), y_mask.shape[0] - y_mask[::-1].argmax())

	# Figure out how big to make the box.
	mask_center = (int(np.mean(x_range)), int(np.mean(y_range)))
	if box_size == None:	
		required_radius = int(np.ceil(np.max([x_range[1] - x_range[0], y_range[1] - y_range[0]])/2))
	else:
		required_radius = box_size

	# Try to make the box.
	cut_x = (mask_center[0] - required_radius, mask_center[0] + required_radius)
	cut_y = (mask_center[1] - required_radius, mask_center[1] + required_radius)
	cut_x = (max(cut_x[0], 0), min(cut_x[1], a_mask.shape[0]))
	cut_y = (max(cut_y[0], 0), min(cut_y[1], a_mask.shape[1]))
	box_size = (cut_x[1] - cut_x[0], cut_y[1] - cut_y[0])
		
	# Make a mask for the chunk and return it.
	square_mask = np.zeros(a_mask.shape).astype('bool')
	square_mask[cut_x[0]:cut_x[1], cut_y[0]:cut_y[1]] = True	
	return (square_mask, box_size)

def lawn_distribution(lawn_folder):
	'''
	Find the distribution of lawn sizes in a folder.
	'''
	def size_object(a_mask):
		'''
		Find the two points farthest apart in a mask and return their distance from each other.
		'''
		locations = np.array(np.ma.nonzero(a_mask)).transpose()
		random_point = locations[0]
		point_a = locations[np.linalg.norm(locations - random_point, axis = 1).argmax()]
		point_b = locations[np.linalg.norm(locations - point_a, axis = 1).argmax()]
		my_distance = np.linalg.norm(point_a - point_b)
		return my_distance
	
	my_lawns = [freeimage.read(lawn_folder + os.path.sep + an_image).astype('bool') for an_image in os.listdir(lawn_folder) if an_image != 'vignette_mask.png']
	my_lawns = [zplib_image_mask.get_largest_object(my_lawn).astype('bool') for my_lawn in my_lawns]
	my_vignette = freeimage.read(lawn_folder + os.path.sep + 'vignette_mask.png').astype('bool')

	lawn_sizes = [size_object(my_lawn) for my_lawn in my_lawns]
	field_size = size_object(my_vignette)
	return (lawn_sizes, field_size)

def select_final_outlines(color_directory, final_directory, working_directory, egg_mode = False):
	'''
	Select the properly filled outlines from color_directory and move them into final_directory along with the appropriate metadata and bright field files from working_directory.
	'''
	mask_ending = 'hmask'
	if egg_mode:
		mask_ending = 'emask'
	for a_subdir in os.listdir(color_directory):
		folderStuff.ensure_folder(final_directory + os.path.sep + a_subdir)
		for an_image in os.listdir(color_directory + os.path.sep + a_subdir):
			if an_image.split(' ')[-1] == 'outline.png':
				destination_mask = final_directory + os.path.sep + a_subdir + os.path.sep + an_image.replace('outline', mask_ending)
				outline_image = freeimage.read(color_directory + os.path.sep + a_subdir + os.path.sep + an_image)
				my_dimensions = len(outline_image.shape)
				if my_dimensions == 2:
					shutil.copyfile(color_directory + os.path.sep + a_subdir + os.path.sep + an_image.replace('outline', mask_ending), destination_mask)
				elif my_dimensions == 3:
					fixed_outline = outline_image[:, :, 0].astype('uint8')
					fixed_outline[fixed_outline > 0] = -1
					freeimage.write(fixed_outline, destination_mask)
				shutil.copyfile(working_directory + os.path.sep + a_subdir + os.path.sep + an_image.replace('outline', 'bf'), destination_mask.replace(mask_ending, 'bf'))
		shutil.copyfile(working_directory + os.path.sep + a_subdir + os.path.sep + 'position_metadata_extended.json', final_directory + os.path.sep + a_subdir + os.path.sep + 'position_metadata_extended.json')
	return

def colored_outlines(a_directory):
	'''
	For each outlined worm in a_directory, convert it to a grayscale 8-bit image with the outline in white and the rest of the image in black.
	'''
	for a_subdir in os.listdir(a_directory):
		for an_image in os.listdir(a_directory + os.path.sep + a_subdir):
			if an_image.split(' ')[-1] == 'bf.png':
				filepath = a_directory + os.path.sep + a_subdir + os.path.sep + an_image
				my_image = freeimage.read(filepath)
				masked_conversion = fill_colored_outline(my_image)
				outline_conversion = fill_colored_outline(my_image, outline_only = True)
				freeimage.write(masked_conversion, filepath.replace('bf', 'hmask'))
				freeimage.write(outline_conversion, filepath.replace('bf', 'outline'))
	return

def colored_color_outlines(a_directory):
	'''
	For each outlined worm in a_directory, convert it to a grayscale 8-bit image with the outline in white and the rest of the image in black.
	'''
	for an_image in os.listdir(a_directory):
		if 'new' not in an_image:
			filepath = a_directory + os.path.sep + an_image
			my_image = freeimage.read(filepath)
			my_red = np.array([237, 28, 36])
			is_red = np.abs(my_image[:, :, :3] - my_red)
			is_red = (is_red.mean(axis = 2) < 1)
			new_image = np.zeros(my_image.shape[:2]).astype('uint8')
			new_image[is_red] = [-1]
			freeimage.write(new_image, filepath.replace('.png', '_new.png'))
	return

def size_histogram(a_directory):
	'''
	return a list of sizes
	'''
	my_files = [a_directory + os.path.sep + a_file for a_file in os.listdir(a_directory)]
	my_images = [freeimage.read(a_file) for a_file in my_files]
	my_masks = [an_image[:, :, 0].astype('bool') for an_image in my_images]
	my_sizes = [np.sum(a_mask) for a_mask in my_masks]
	return my_sizes

def plate_sizes_12_day():
	'''	
	Return a bunch of sizes in pixels.
	'''
	# Raw sizes in pixels.	
	size_series = pd.Series([9907, 7385, 5596, 6668, 7250, 2529, 3986, 4285, 2627, 9019, 3833, 7223, 6274, 3398, 5099, 2925, 7349, 4975, 3940, 5809, 6507, 3479])	

	# Points are 2 mm apart based on micrometer.
	point_a = np.array((294, 364)) 
	point_b = np.array((843, 388))
	pixel_distance = np.linalg.norm(point_a - point_b)
	
	# Do the conversion to mm^2.
	pixel_to_mm = (2/pixel_distance)**2
	size_series = size_series*pixel_to_mm
	return size_series

def egg_outlines(a_directory):
	'''
	For each outlined egg mask in a_directory, convert it to a grayscale 8-bit image with the outline in white and the rest of the image in black.
	'''
	for a_subdir in os.listdir(a_directory):
		for an_image in os.listdir(a_directory + os.path.sep + a_subdir):
			if an_image.split(' ')[-1] == 'bf.png':
				filepath = a_directory + os.path.sep + a_subdir + os.path.sep + an_image
				my_image = freeimage.read(filepath)
				masked_conversion = fill_colored_outline(my_image, egg_mode = True)
				outline_conversion = fill_colored_outline(my_image, egg_mode = True, outline_only = True)
				freeimage.write(masked_conversion, filepath.replace('bf', 'emask'))
				freeimage.write(outline_conversion, filepath.replace('bf', 'outline'))
	return


def fill_colored_outline(an_image, outline_only = False, egg_mode = False):
	'''
	Fill out a colored outline and return a mask.
	'''
	# Get rid of the alpha channel.
	an_image = an_image.copy()
	an_image[:, :, 3] = 0
	
	# Get pixels with disproportionately more red.	
	my_mask = np.abs(np.max(an_image[:, :, 1:], axis = 2).astype('float64') - an_image[:, :, 0].astype('float64')).astype('uint16') > 0
	if not egg_mode:
		my_mask = zplib_image_mask.get_largest_object(my_mask)	
	
	# Fill holes.
	if not outline_only:		
		my_mask = zplib_image_mask.fill_small_area_holes(my_mask, 300000).astype('uint8')
	my_mask = my_mask.astype('uint8')
	my_mask[my_mask >= 1] = -1 
	return my_mask
		
def random_select_ages(data_dir):
	'''
	Randomly select 50 worms from each 3-day age range to trace.
	'''
	random_bins = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
	my_bins = [[] for i in range(5)]
	for a_worm in [a_worm for a_worm in os.listdir(data_dir) if os.path.isdir(data_dir + os.path.sep + a_worm)]:		
		with open(data_dir + os.path.sep + a_worm + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
			position_metadata = json.loads(read_file.read())
		death_time = position_metadata[0]['death']
		death_age = [position_metadata[i]['age'] for i in range(0, len(position_metadata)) if position_metadata[i]['timepoint'] == death_time][0]
		for a_position in position_metadata:
			if 0 < a_position['age'] <= death_age:
				for i in range(0, len(random_bins)):
					a_bin = random_bins[i] 
					if a_bin[0] < a_position['age']/24 <= a_bin[1]:
						my_bins[i].append(data_dir + os.path.sep + a_worm + os.path.sep + a_position['timepoint'] + ' bf.png')
	final_bins = [random.sample(my_bins[i], min(50, len(my_bins[i]))) for i in range(0, len(my_bins))]
	final_list = []
	for a_list in final_bins:
		final_list.extend(a_list)
	return final_list

def random_select_egg_ages(data_dir):
	'''
	Randomly select 50 worms from each 3-day age range to trace.
	'''
	random_bins = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
	my_bins = [[] for i in range(5)]
	for a_worm in [a_worm for a_worm in os.listdir(data_dir) if os.path.isdir(data_dir + os.path.sep + a_worm)]:		
		with open(data_dir + os.path.sep + a_worm + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
			position_metadata = json.loads(read_file.read())
		death_time = position_metadata[0]['death']
		death_age = [position_metadata[i]['egg_age'] for i in range(0, len(position_metadata)) if position_metadata[i]['timepoint'] == death_time][0]
		for a_position in position_metadata:
			if 0 < a_position['egg_age'] <= death_age:
				for i in range(0, len(random_bins)):
					a_bin = random_bins[i] 
					if a_bin[0] < a_position['egg_age']/24 <= a_bin[1]:
						my_bins[i].append(data_dir + os.path.sep + a_worm + os.path.sep + a_position['timepoint'] + ' bf.png')
	final_bins = [random.sample(my_bins[i], min(200, len(my_bins[i]))) for i in range(0, len(my_bins))]
	final_list = []
	for a_list in final_bins:
		final_list.extend(a_list)
	return final_list
	
def random_select(data_dir, image_number = 200):
	'''
	Randomly select image_number images from subdirectories of data_dir for tracing.
	'''
	one_bin = []
	for a_worm in [a_worm for a_worm in os.listdir(data_dir) if os.path.isdir(data_dir + os.path.sep + a_worm)]:		
		with open(data_dir + os.path.sep + a_worm + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
			position_metadata = json.loads(read_file.read())
		death_time = position_metadata[0]['death']
		death_age = [position_metadata[i]['age'] for i in range(0, len(position_metadata)) if position_metadata[i]['timepoint'] == death_time][0]
		for a_position in position_metadata:
			if 0 < a_position['age'] <= death_age:
				one_bin.append(data_dir + os.path.sep + a_worm + os.path.sep + a_position['timepoint'] + ' bf.png')
	one_bin = random.sample(one_bin, image_number)
	return one_bin

def select_daily(data_dir):
	'''
	For each worm, select daily images to trace after hatch.
	'''
	selected_images = []
	for a_worm in os.listdir(data_dir):	
		if os.path.isdir(data_dir + os.path.sep + a_worm) and a_worm != 'calibrations' and a_worm != 'measured_health':
			
			with open(data_dir + os.path.sep + a_worm + os.path.sep + 'position_metadata_extended.json', 'r') as read_file:
				position_metadata = json.loads(read_file.read())
			death_time = position_metadata[0]['death']
			death_age = [position_metadata[i]['age'] for i in range(0, len(position_metadata)) if position_metadata[i]['timepoint'] == death_time][0]/24
			time_ages = np.array([position_metadata[i]['age'] for i in range(0, len(position_metadata))])/24
			for i in range(1, int(death_age)):
				day_timepoint = np.argmin(np.abs(time_ages - i))
				selected_images.append(data_dir + os.path.sep + a_worm + os.path.sep + position_metadata[day_timepoint]['timepoint'] + ' bf.png')
	return selected_images

def remove_already_have(my_list, human_dir):
	'''
	Remove ones that we already have.
	'''
	search_list = []	
	for subdir in [a_subdir for a_subdir in os.listdir(human_dir) if os.path.isdir(human_dir + os.path.sep + a_subdir)]:
		for a_file in os.listdir(human_dir + os.path.sep + subdir):
			search_name = subdir + os.path.sep + a_file.replace('hmask', 'bf')
			search_list.append(search_name)
	new_list = [im for im in my_list if os.path.sep.join(im.split(os.path.sep)[-2:]) not in search_list]
	return new_list	

def pick_masks_to_trace(working_dir, human_dir, temporary_dir):
	'''
	Copy the bright-field images to trace from working_dir to temporary_dir.
	'''
	my_masks = random_select_egg_ages(working_dir)
	my_masks = remove_already_have(my_masks, human_dir)
	for a_mask in my_masks:
		base_folder = temporary_dir
		subdirectory_folder = os.path.sep.join(a_mask.split(os.path.sep)[:-1]).replace(working_dir, temporary_dir)
		folderStuff.ensure_folder(base_folder)
		folderStuff.ensure_folder(subdirectory_folder)
		if os.path.isfile(a_mask):
			shutil.copy(a_mask, a_mask.replace(working_dir, temporary_dir))
	return

def totally_random(working_dir, human_dir, temporary_dir, image_number = 200):
	'''
	Select image_number masks from working_dir without regard for age bins.
	'''
	my_masks = random_select(working_dir, image_number)
	my_masks = remove_already_have(my_masks, human_dir)
	for a_mask in my_masks:
		base_folder = temporary_dir
		subdirectory_folder = os.path.sep.join(a_mask.split(os.path.sep)[:-1]).replace(working_dir, temporary_dir)
		folderStuff.ensure_folder(base_folder)
		folderStuff.ensure_folder(subdirectory_folder)
		if os.path.isfile(a_mask):
			shutil.copy(a_mask, a_mask.replace(working_dir, temporary_dir))
	return

def pick_full_masks(working_dir, human_dir, temporary_dir):
	'''
	Copy the bright-field images to trace (giving up on automated wormFinding) from working_dir to temporary_dir.
	'''
	my_masks = select_daily(working_dir)
	my_masks = remove_already_have(my_masks, human_dir)
	for a_mask in my_masks:
		base_folder = temporary_dir
		subdirectory_folder = os.path.sep.join(a_mask.split(os.path.sep)[:-1]).replace(working_dir, temporary_dir)
		folderStuff.ensure_folder(base_folder)
		folderStuff.ensure_folder(subdirectory_folder)
		shutil.copyfile(a_mask, a_mask.replace(working_dir, temporary_dir))
	return

def cut_out_box(image_array, center, box_size):
	'''
	Cut out the portion of image_array (which is 2-dimensional/grayscale) centered at the pixel located at center, of size box_size. Note that box_size should be odd in each dimension for this to work properly, keeping the center pixel perfectly centered.
	'''
	box_radii = np.array(((box_size[0]-1)/2, (box_size[1]-1)/2))
	my_box = image_array[center[0] - box_radii[0]: center[0] + box_radii[0] + 1, center[1] - box_radii[1]: center[1] + box_radii[1] + 1]
	return my_box

def color_dots(image_paths, save_paths, color_dot_location_lists):
	'''
	Draws colored dots on images from image_paths according to color_dot_location_lists, and saves them out to save_paths.
	'''
	color_dot_location_lists = np.round(color_dot_location_lists)
	for i in range(0, len(image_paths)):
		image_array = freeimage.read(image_paths[i])
		print('Drawing dots for ' + image_paths[i] + '.')
		
		if len(image_array.shape) == 2:
			my_width, my_height = image_array.shape
			color_array = np.empty((my_width, my_height, 3), dtype = image_array.dtype)
			color_array[:, :, 0] = image_array.copy()
			color_array[:, :, 1] = image_array.copy()
			color_array[:, :, 2] = image_array.copy()
	
		elif len(image_array.shape) == 3:
			color_array = image_array
		
		color_array[color_dot_location_lists[0][i][0], color_dot_location_lists[0][i][1], :] = [0, 0, 0]
		if len(color_dot_location_lists) > 1:		
			color_array[color_dot_location_lists[1][i][0], color_dot_location_lists[1][i][1], :] = [0, 0, 0]
		if len(color_dot_location_lists) > 2:		
			color_array[color_dot_location_lists[2][i][0], color_dot_location_lists[2][i][1], :] = [0, 0, 0]

		color_array[color_dot_location_lists[0][i][0], color_dot_location_lists[0][i][1], 0] = -1
		if len(color_dot_location_lists) > 1:		
			color_array[color_dot_location_lists[1][i][0], color_dot_location_lists[1][i][1], 1] = -1
		if len(color_dot_location_lists) > 2:		
			color_array[color_dot_location_lists[2][i][0], color_dot_location_lists[2][i][1], 2] = -1

		freeimage.write(color_array, save_paths[i])
	return
	
def renormalize_image(an_image):
	'''
	Rescale an image's brightness to make viewing on windows better.
	'''
	normalized = an_image.copy()
	if normalized.dtype == 'uint16':
		im_min = np.min(normalized)
		im_max = np.max(normalized)
		normalized = np.float64(normalized-im_min)/(im_max-im_min)
		normalized = normalized*(2**16-50)		
		normalized = normalized.astype('uint16')
	if normalized.dtype == 'uint8':
		im_min = np.min(normalized)
		im_max = np.max(normalized)
		normalized = np.float64(normalized-im_min)/(im_max-im_min)
		normalized = normalized*(2**8-1)		
		normalized = normalized.astype('uint8')
	return normalized
	
def archive_human_masks(human_directory, new_directory, work_directory):
	'''
	For a directory of hand-drawn masks, mask out everything in the accompanying bright-field file except for the worm itself and a 100-pixel surrounding area to save disk space. Also, re-compress all images to maximize compression and space efficiency.
	'''
	for a_subdir in os.listdir(human_directory):
		if os.path.isdir(human_directory + os.path.sep + a_subdir):
			folderStuff.ensure_folder(new_directory + os.path.sep + a_subdir)
			for a_file in os.listdir(human_directory + os.path.sep + a_subdir):
				if a_file.split(' ')[-1] == 'hmask.png':
					if not os.path.isfile(new_directory + os.path.sep + a_subdir + os.path.sep + a_file):
						print('Up to ' + a_subdir + ' ' + a_file + '.')
						my_stem = a_file.split(' ')[0]
						my_mask = freeimage.read(human_directory + os.path.sep + a_subdir + os.path.sep + my_stem + ' ' + 'hmask.png')
						bf_path = human_directory + os.path.sep + a_subdir + os.path.sep + my_stem + ' ' + 'bf.png'
						if os.path.isfile(bf_path):
							my_image = freeimage.read(bf_path)
						else:
							my_image = freeimage.read(bf_path.replace(human_directory, work_directory))
						area_mask = my_mask.copy().astype('bool')
						distance_from_mask = scipy.ndimage.morphology.distance_transform_edt(np.invert(area_mask)).astype('uint16')
						area_mask[distance_from_mask > 0] = True
						area_mask[distance_from_mask > 100] = False
						my_image[np.invert(area_mask)] = False
						freeimage.write(my_image, new_directory + os.path.sep + a_subdir + os.path.sep + my_stem + ' ' + 'bf.png', flags = freeimage.IO_FLAGS.PNG_Z_BEST_COMPRESSION)					
						freeimage.write(my_mask, new_directory + os.path.sep + a_subdir + os.path.sep + my_stem + ' ' + 'hmask.png', flags = freeimage.IO_FLAGS.PNG_Z_BEST_COMPRESSION)					
				elif a_file.split('.')[-1] == 'json':					
					shutil.copyfile(human_directory + os.path.sep + a_subdir + os.path.sep + a_file, new_directory + os.path.sep + a_subdir + os.path.sep + a_file)
	return
	
def main():
	return

if __name__ == "__main__":
	main()
