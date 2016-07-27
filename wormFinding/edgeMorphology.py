# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:41:38 2016

@author: Willie
"""

import os
import skimage.filters
import skimage.feature
import skimage.morphology
import scipy.ndimage
import numpy as np
import concurrent.futures
import multiprocessing
import cv2
import json 

import freeimage
import zplib.image.mask as zplib_image_mask

import basicOperations.imageOperations as imageOperations

def read_corrected_bf(image_file, movement_key = ''):
	'''
	Read in an image at time_point and properly correct it for flatfield and metering.
	'''
	time_point = image_file.split(os.path.sep)[-1].split(' ')[0]
	raw_image = freeimage.read(image_file)		
	flatfield_image = freeimage.read(os.path.sep.join(image_file.split(os.path.sep)[:-2]) + os.path.sep + 'calibrations' + os.path.sep + time_point + ' ' + 'bf_flatfield.tiff')
	with open(os.path.sep.join(image_file.split(os.path.sep)[:-2]) + os.path.sep + 'experiment_metadata.json', 'r') as read_file:
		metadata = json.loads(read_file.read())
	time_reference = metadata['brightfield metering'][time_point]['ref_intensity']
	corrected_image = raw_image*flatfield_image	
	corrected_image = corrected_image / time_reference * 11701.7207031
	corrected_image = corrected_image.astype('uint16')
	return corrected_image	

def make_mega_lawn(worm_subdirectory, super_vignette):
	'''
	Make a mega lawn mask for use with all images of a worm. This avoids problems with detecting the relatively faint edge of the lawn, which depends on the exact focus plane and the brightness of the lamp.
	'''
	# Parallelized edge detection.
	my_bf_files = [worm_subdirectory + os.path.sep + a_bf for a_bf in os.listdir(worm_subdirectory) if a_bf[-6:] == 'bf.png']
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	chunk_size = int(np.ceil(len(my_bf_files)/my_workers))
	bf_chunks = [my_bf_files[x:x + chunk_size] for x in range(0, len(my_bf_files), chunk_size)]
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		chunk_masks = [executor.submit(lawn_maker, bf_chunks[i], super_vignette.copy()) for i in range(0, len(bf_chunks))]
	concurrent.futures.wait(chunk_masks)
	chunk_masks = [a_job.result() for a_job in chunk_masks]
	
	# Make mega lawn from edge detection.
	mega_lawn = np.max(np.array(chunk_masks), axis = 0)
	mega_lawn = scipy.ndimage.morphology.binary_fill_holes(mega_lawn).astype('bool')
	mega_lawn = zplib_image_mask.get_largest_object(mega_lawn).astype('uint8')
	mega_lawn[mega_lawn > 0] = -1	
	
	# Parallelized thresholding.
	with concurrent.futures.ProcessPoolExecutor(max_workers = my_workers) as executor:
		chunk_masks = [executor.submit(alternate_lawn_maker, bf_chunks[i], super_vignette.copy()) for i in range(0, len(bf_chunks))]
	concurrent.futures.wait(chunk_masks)
	chunk_masks = [a_job.result() for a_job in chunk_masks]			
		
	# Make alternative mega lawn from thresholding intensity.
	alt_mega_lawn = np.max(np.array(chunk_masks), axis = 0)
	alt_mega_lawn = scipy.ndimage.morphology.binary_fill_holes(alt_mega_lawn).astype('bool')
	alt_mega_lawn = zplib_image_mask.get_largest_object(alt_mega_lawn).astype('uint8')
	alt_mega_lawn[alt_mega_lawn > 0] = -1
	
	# Select proper mega lawn.
	if np.bincount(np.ndarray.flatten(mega_lawn))[-1] < 0.8*np.bincount(np.ndarray.flatten(alt_mega_lawn))[-1]:
		mega_lawn = alt_mega_lawn
	freeimage.write(mega_lawn, worm_subdirectory + os.path.sep + 'great_lawn.png')
	return mega_lawn

	
def alternate_lawn_maker(bf_files, super_vignette):
	'''
	Make a lawn just by intensity when for some reason the normal lawn maker fails...
	'''
	lawn_masks = []
	for a_bf_file in bf_files:
		renormalized_image = read_corrected_bf(a_bf_file)
		lawn_cutoff = np.percentile(renormalized_image, 5)
		lawn_mask = np.zeros(renormalized_image.shape).astype('bool')
		lawn_mask[renormalized_image < lawn_cutoff] = True
		ultra_vignette = scipy.ndimage.morphology.binary_erosion(super_vignette, iterations = 10)
		lawn_mask[np.invert(ultra_vignette)] = False 
		lawn_mask = scipy.ndimage.morphology.binary_dilation(lawn_mask, iterations = 3)
		lawn_mask = scipy.ndimage.morphology.binary_fill_holes(lawn_mask)
		lawn_mask = zplib_image_mask.get_largest_object(lawn_mask).astype('bool')
		lawn_mask = scipy.ndimage.morphology.binary_erosion(lawn_mask, iterations = 3)
		lawn_mask = zplib_image_mask.get_largest_object(lawn_mask).astype('bool')
		lawn_masks.append(lawn_mask)
	lawn_mask = np.max(np.array(lawn_masks), axis = 0)
	return lawn_mask

def lawn_maker(bf_files, super_vignette):
	'''
	Find the bacterial lawn in one worm image.
	'''
	lawn_masks = []
	for a_bf_file in bf_files:
		# Prepare a worm image for use in lawn-finding.
		renormalized_image = freeimage.read(a_bf_file)
		renormalized_image = cv2.medianBlur(renormalized_image, 3)
		renormalized_image = imageOperations.renormalize_image(renormalized_image)
		
		# Remove extraneous edges and out-of-lawn junk by finding the lawn and also applying an "ultra-vignette" mask.
		ultra_vignette = scipy.ndimage.morphology.binary_erosion(super_vignette, iterations = 10)
		my_edges = skimage.feature.canny(renormalized_image, sigma = 0.02)
		my_edges[np.invert(ultra_vignette)] = False 
		my_edges = scipy.ndimage.morphology.binary_dilation(my_edges, iterations = 10)
		my_lawn = scipy.ndimage.morphology.binary_fill_holes(my_edges)
		try:
			my_lawn = zplib_image_mask.get_largest_object(my_lawn).astype('bool')
			my_lawn = scipy.ndimage.morphology.binary_erosion(my_lawn, iterations = 10)
			my_lawn = zplib_image_mask.get_largest_object(my_lawn).astype('bool')
		except:
			my_lawn = np.zeros(my_lawn.shape).astype('bool')
		lawn_masks.append(my_lawn)
	my_lawn = np.max(np.array(lawn_masks), axis = 0)
	return my_lawn

def backup_worm_find(my_edges, bg_center):
	'''
	If my edge detection doesn't find something close enough to the background mask, take more drastic action.
	'''
	def worm_candidate(my_edges, center_pixel, iterations):
		'''
		Find a candidate worm.
		'''
		harder_eggs = scipy.ndimage.morphology.binary_dilation(my_edges, iterations = iterations)
		harder_eggs = scipy.ndimage.morphology.binary_fill_holes(harder_eggs)
		harder_eggs = scipy.ndimage.morphology.binary_erosion(harder_eggs, iterations = iterations)
		if np.bincount(np.ndarray.flatten(harder_eggs)).shape[0] == 1:
			harder_eggs[center_pixel[0], center_pixel[1]] = True
		the_worm = zplib_image_mask.get_largest_object(harder_eggs).astype('bool')
		return the_worm
	
	# Get the eggs that have been edge-detected as closed shapes.
	center_pixel = (my_edges.shape[0]//2, my_edges.shape[1]//2)
	easy_eggs = scipy.ndimage.morphology.binary_fill_holes(my_edges)
	if np.bincount(np.ndarray.flatten(easy_eggs)).shape[0] == 1:
		easy_eggs[center_pixel[0], center_pixel[1]] = True
	easy_worm = zplib_image_mask.get_largest_object(easy_eggs).astype('bool')

	# Get the more difficult eggs and the worm itself.
	worm_candidates = [easy_worm]
	worm_candidates.extend([worm_candidate(my_edges, center_pixel, i) for i in range(3, 11)])
	worm_centers = np.array([np.mean(np.array(np.where(worm_candidate > 0)), axis = 1) for worm_candidate in worm_candidates])
	worm_distances = [np.linalg.norm(worm_center - bg_center) for worm_center in worm_centers]

	the_worm = None
	done_yet = False	
	for i in range(0, len(worm_candidates)):
		if not done_yet:
			if worm_distances[i] < 20:
				done_yet = True
				the_worm = worm_candidates[i]
				the_worm = scipy.ndimage.morphology.binary_fill_holes(the_worm)	
	return the_worm

def find_eggs_worm(worm_image, super_vignette, mega_lawn, worm_sigma):
	'''
	Solve the segmentation problem that has been giving me hives for months in a weekend...
	'''
	# Prepare for edge detection by renormalizing and denoising image.
	worm_image = worm_image.copy()
	worm_image = cv2.medianBlur(worm_image, 3)
	worm_image = imageOperations.renormalize_image(worm_image)
	renormalized_image = worm_image

	# Find non-lawn and vignetting artifacts, and prepare to remove lawn and vignette edges.
	ultra_vignette = scipy.ndimage.morphology.binary_erosion(super_vignette, iterations = 10)
	
	# Do the actual edge detection in search of the worm.
	my_edges = skimage.feature.canny(renormalized_image, sigma = worm_sigma)
	my_edges[np.invert(ultra_vignette)] = False 
	my_edges[np.invert(mega_lawn)] = False
	
	# Get the eggs that have been edge-detected as closed shapes.
	center_pixel = (worm_image.shape[0]//2, worm_image.shape[1]//2)
	easy_eggs = scipy.ndimage.morphology.binary_fill_holes(my_edges)
	(labels, region_indices, areas) = zplib_image_mask.get_areas(easy_eggs)
	keep_labels = areas > (1/(0.7**2))*270
	keep_labels = np.concatenate(([0], keep_labels))
	easy_eggs = keep_labels[labels].astype('bool')
	if np.bincount(np.ndarray.flatten(easy_eggs)).shape[0] == 1:
		easy_eggs[center_pixel[0], center_pixel[1]] = True	
	easy_worm = zplib_image_mask.get_largest_object(easy_eggs).astype('bool')

	# Get the more difficult eggs and the worm itself.
	harder_eggs = scipy.ndimage.morphology.binary_dilation(my_edges, iterations = 3)
	harder_eggs = scipy.ndimage.morphology.binary_fill_holes(harder_eggs)
	harder_eggs = scipy.ndimage.morphology.binary_erosion(harder_eggs, iterations = 3)
	harder_eggs[easy_eggs] = False
	(labels, region_indices, areas) = zplib_image_mask.get_areas(harder_eggs)
	keep_labels = areas > (1/(0.7**2))*270
	keep_labels = np.concatenate(([0], keep_labels))
	harder_eggs = keep_labels[labels].astype('bool')
	if np.bincount(np.ndarray.flatten(harder_eggs)).shape[0] == 1:
		harder_eggs[center_pixel[0], center_pixel[1]] = True
	the_worm = zplib_image_mask.get_largest_object(harder_eggs).astype('bool')
	if the_worm[the_worm].shape[0] < easy_worm[easy_worm].shape[0]:
		the_worm = easy_worm

	if the_worm[the_worm].shape[0] > 200000:
		lawn_edges = skimage.feature.canny(renormalized_image, sigma = 0.05)
		lawn_edges[np.invert(ultra_vignette)] = False 
		lawn_edges = scipy.ndimage.morphology.binary_dilation(lawn_edges, iterations = 10)
		my_lawn = scipy.ndimage.morphology.binary_fill_holes(lawn_edges)
		my_lawn = zplib_image_mask.get_largest_object(my_lawn).astype('bool')
		my_lawn = scipy.ndimage.morphology.binary_erosion(my_lawn, iterations = 10)
		lawn_transform = scipy.ndimage.morphology.distance_transform_edt(my_lawn)
		my_lawn_edge = np.zeros(my_lawn.shape).astype('bool')
		my_lawn_edge[lawn_transform > 0] = True 	
		my_lawn_edge[lawn_transform > 6] = False 	
		
		# Do the actual edge detection in search of the worm.
		my_edges = skimage.feature.canny(renormalized_image, sigma = 2.0)
		my_edges[np.invert(ultra_vignette)] = False 
		my_edges[np.invert(mega_lawn)] = False
		my_edges[my_lawn_edge] = False
		
		# Get the eggs that have been edge-detected as closed shapes.
		center_pixel = (worm_image.shape[0]//2, worm_image.shape[1]//2)
		easy_eggs = scipy.ndimage.morphology.binary_fill_holes(my_edges)
		(labels, region_indices, areas) = zplib_image_mask.get_areas(easy_eggs)
		keep_labels = areas > (1/(0.7**2))*270
		keep_labels = np.concatenate(([0], keep_labels))
		easy_eggs = keep_labels[labels].astype('bool')
		if np.bincount(np.ndarray.flatten(easy_eggs)).shape[0] == 1:
			easy_eggs[center_pixel[0], center_pixel[1]] = True	
		easy_worm = zplib_image_mask.get_largest_object(easy_eggs).astype('bool')
		
		# Get the more difficult eggs and the worm itself.
		harder_eggs = scipy.ndimage.morphology.binary_dilation(my_edges, iterations = 3)
		harder_eggs = scipy.ndimage.morphology.binary_fill_holes(harder_eggs)
		harder_eggs = scipy.ndimage.morphology.binary_erosion(harder_eggs, iterations = 3)
		harder_eggs[easy_eggs] = False
		(labels, region_indices, areas) = zplib_image_mask.get_areas(harder_eggs)
		keep_labels = areas > (1/(0.7**2))*270
		keep_labels = np.concatenate(([0], keep_labels))
		harder_eggs = keep_labels[labels].astype('bool')
		if np.bincount(np.ndarray.flatten(harder_eggs)).shape[0] == 1:
			harder_eggs[center_pixel[0], center_pixel[1]] = True
		the_worm = zplib_image_mask.get_largest_object(harder_eggs).astype('bool')
		if the_worm[the_worm].shape[0] < easy_worm[easy_worm].shape[0]:
			the_worm = easy_worm		

	# Clean up worm and eggs.
	the_worm = scipy.ndimage.morphology.binary_fill_holes(the_worm)	
	harder_eggs[the_worm] = False
	easy_eggs[the_worm] = False
	
	# Combine the two groups of eggs.
	all_eggs = np.zeros(harder_eggs.shape).astype('bool')
	all_eggs[harder_eggs] = True
	all_eggs[easy_eggs] = True	
	return (all_eggs, the_worm, my_edges)

def main():
	return

if __name__ == "__main__":
	main()
