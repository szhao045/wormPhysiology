# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:07:28 2015

@author: Willie
"""

import os
import pickle
import skimage
import skimage.morphology
import numpy as np
import scipy
import pandas as pd

import pyagg
import freeimage
from zplib import pca as zplib_pca
from zplib.curve import interpolate as zplib_interpolate
from zplib.curve import geometry as zplib_geometry
from zplib.image import mask as zplib_image_mask

def normalize_worm(my_spine, final_angle = None, final_scale = None, return_scale_factor = False):
	'''
	Rotates and rescales a spine so that its long axis is at final_angle and its scale factor is at final_scale.
	'''
	normalized_spine = my_spine.copy()	
	
	# Normalize the scale.
	if final_scale != None:	
		spine_vectors = normalized_spine[1:] - normalized_spine[:-1]
		spine_angles = np.array([np.arctan2(spine_vector[1], spine_vector[0]) for spine_vector in spine_vectors])
		scale_factor = np.mean(np.array([np.linalg.norm(spine_vector) for spine_vector in spine_vectors]))
		
		spine_vectors_x = [np.cos(spine_angle) for spine_angle in spine_angles]
		spine_vectors_y = [np.sin(spine_angle) for spine_angle in spine_angles]
		spine_vectors = [np.array([spine_vectors_x[i], spine_vectors_y[i]]) for i in range(0, len(spine_vectors_x))]	
		scaled_spine_vectors = np.array([final_scale*a_vector/np.linalg.norm(a_vector) for a_vector in spine_vectors])
		normalized_spine = [np.array([0, 0])]
		for i in range(0, len(scaled_spine_vectors)):
			normalized_spine.append(normalized_spine[-1] + scaled_spine_vectors[i])
		normalized_spine = np.array(normalized_spine)
	
	# Normalize the long axis angle.
	if final_angle != None:
		my_head = normalized_spine[0, :]
		my_tail = normalized_spine[-1, :]
		long_axis = my_tail - my_head
		long_angle = np.arctan2(long_axis[1], long_axis[0])
		rotation_matrix = np.array([[np.cos(-long_angle + final_angle), -np.sin(-long_angle + final_angle)], [np.sin(-long_angle + final_angle), np.cos(-long_angle + final_angle)]])
		normalized_spine = np.array([rotation_matrix.dot(spine_vector) for spine_vector in normalized_spine])
	
	if return_scale_factor:
		return (normalized_spine, scale_factor)
	return normalized_spine
	
def train_PCs(training_data_dir, age_range = None, n_points = 100):
	'''
	Takes worm masks and metadata from subdirectories in training_data_dir and generates principal components from the set of 99 angles tangent to the worms' outlines.
	'''	
	training_subdirs = [os.path.join(training_data_dir, subdir) for subdir in os.listdir(training_data_dir) if os.path.isdir(os.path.join(training_data_dir, subdir))]
	endings = ['bf.png', 'mask.png', 'metadata.pickle']	
	
	all_shapes = []
	scale_factors = []
	width_list = []
	for subdir in training_subdirs:
		total_files = os.listdir(subdir)
		data_points = [' '.join(a_file.split('.')[0].split(' ')[0:-1]) for a_file in total_files if a_file.split('.')[-1] == 'pickle']
		for a_point in data_points:
			my_metadata = pickle.load(open(subdir + os.path.sep + a_point + ' ' + endings[2], 'rb'))
			if age_range == None or age_range[0] <= my_metadata['age_days'] <= age_range[1]:
				spine_tck = my_metadata['spine_tck']
				width_tck = my_metadata['width_tck']
				my_spine = zplib_interpolate.spline_interpolate(spine_tck, n_points)
				widths = zplib_interpolate.spline_interpolate(width_tck, n_points)
				(my_spine, scale_factor) = normalize_worm(my_spine, final_scale = 1, final_angle = 0, return_scale_factor = True)
				scale_factors.append(scale_factor)
				width_list.append(widths)
				all_shapes.append(np.ndarray.flatten(my_spine))
				
	class a_PCA():
		def __init__(self, mean, pcs, norm_pcs, variances, positions, norm_positions, scale_factors):
			self.mean = mean
			self.pcs = pcs
			self.norm_pcs = norm_pcs
			self.variances = variances
			self.positions = positions
			self.norm_positions = norm_positions
			self.components_ = norm_pcs
			self.mean_scale = np.mean(scale_factors)
			
	mean, pcs, norm_pcs, variances, positions, norm_positions = zplib_pca.pca(all_shapes)			
	width_mean, width_pcs, width_norm_pcs, width_variances, width_positions, width_norm_positions = zplib_pca.pca(width_list)			
	my_PCA = a_PCA(mean, pcs, norm_pcs, variances, positions, norm_positions, scale_factors)
	width_PCA = a_PCA(width_mean, width_pcs, width_norm_pcs, width_variances, width_positions, width_norm_positions, scale_factors)
	print('Finished training spine points PCs.')
	return (my_PCA, width_PCA)

def deviation_projection(a_worm, my_PCA):
	'''
	Finds the deviation of a_worm from the mean worm in terms of PCs.
	'''
	def project_onto_components(a_worm, my_PCA):
		'''
		Projects a_worm on to my_PCA.components_ and returns an array of the components.
		'''
		def vector_project(my_vector, a_direction):
			'''
			Computes the vector projection of my_vector on to a_direction.
			'''
			my_vector = np.array(my_vector)
			a_direction = np.array(a_direction)
			unit_direction = a_direction/np.linalg.norm(a_direction)
			component_length = np.linalg.norm(a_direction)
			if component_length > 0:
				magnitude_result = np.array(my_vector).dot(a_direction)/component_length
			else:
				magnitude_result = 0
			final_result = magnitude_result*unit_direction
			return final_result
	
		def scalar_project(my_vector, a_direction):
			'''
			Computes the scalar projection of my_vector on to a_direction.
			'''
			my_vector = np.array(my_vector)
			a_direction = np.array(a_direction)
			component_length = np.linalg.norm(a_direction)
			if component_length > 0:
				magnitude_result = np.array(my_vector).dot(a_direction)/component_length
			else:
				magnitude_result = 0
			return magnitude_result
	
		my_component_weights = np.zeros(my_PCA.components_.shape[0])
		for i in range(0, my_PCA.components_.shape[0]):
			projection_len = scalar_project(a_worm, my_PCA.components_[i])
			component_length = np.linalg.norm(my_PCA.components_[i])
			if component_length > 0:
				my_component_weights[i] = np.divide(projection_len, component_length)
			else: 
				my_component_weights[i] = 0
		return np.nan_to_num(my_component_weights)	
	
	projected_worm = project_onto_components(a_worm, my_PCA)
	projected_mean_worm = project_onto_components(my_PCA.mean, my_PCA)
	deviation = projected_worm - projected_mean_worm
	return deviation

def generate_worm(my_PCA, width_PCA, my_parameters):
	'''
	Generates a specific worm from parameters in my_parameters and the PCA object my_PCA.
	'''
	mean_worm = my_PCA.mean	
	mean_widths = width_PCA.mean
	pc_weights = my_parameters[:-4-5]

	pc_weights = np.array(pc_weights)
	all_weights = np.hstack([pc_weights, np.zeros(my_PCA.components_.shape[0] -  pc_weights.shape[0])])	
	pc_deviation = all_weights.dot(my_PCA.components_)
	my_worm = mean_worm	+ pc_deviation

	my_worm = my_worm.reshape(my_worm.shape[0]/2, 2)
	my_spine = normalize_worm(my_worm, final_angle = my_parameters[-1], final_scale = my_parameters[-4])
	my_spine = np.ndarray.flatten(my_spine)
	
	width_pc_weights = my_parameters[-5-4:-4]
	all_width_weights = np.hstack([width_pc_weights, np.zeros(width_PCA.components_.shape[0] - width_pc_weights.shape[0])])	
	width_deviation = all_width_weights.dot(width_PCA.components_)	
	my_worm_widths = mean_widths + width_deviation
	return (my_spine, my_worm_widths)

def rasterize_worm(my_worm_spine, my_worm_widths, my_parameters, canvas_shape):
	'''
	Takes the points of the spine of a worm given by my_worm_spine and widths given by my_worm_widths, parameters in my_parameters, and the overall shape of the canvas/picture given by canvas_shape, and creates a generated worm mask.
	'''
	center_of_mass = my_parameters[-3:-1]
	my_worm_spine = my_worm_spine.reshape(my_worm_spine.shape[0]/2, 2)
	my_array = np.zeros(canvas_shape).astype('uint8')
	my_canvas = pyagg.ndarray_canvas_g8(my_array)

	perpendiculars = zplib_geometry.find_polyline_perpendiculars(my_worm_spine)
	offsets = perpendiculars * my_worm_widths[:, np.newaxis]

	left = my_worm_spine + offsets
	right = my_worm_spine - offsets
	outline = np.concatenate([left, right[::-1]], axis=0)
	my_worm = outline

	# Center the worm where it belongs. The center of mass is a rough number, since it is a parameter which must be fit anyway.
	my_worm[:, 0] = my_worm[:, 0] - np.mean(my_worm, axis = 0)[0] + center_of_mass[0]*canvas_shape[0]
	my_worm[:, 1] = my_worm[:, 1] - np.mean(my_worm, axis = 0)[1] + center_of_mass[1]*canvas_shape[1]

	yx_worm = np.zeros(my_worm.shape)
	yx_worm[:, 0] = my_worm[:, 1]
	yx_worm[:, 1] = my_worm[:, 0]

	my_canvas.draw_polygon(yx_worm, fill = True, aa = False)
	my_array = my_array.astype('uint8')
	my_array[my_array > 0] = -1	
	return my_array

def pc_deviations_picture(my_PCA_stuff, standard_mask, worm_maker, worm_drawer, vertical_combine, horizontal_combine, output_file):
	'''
	Takes a my_PCA_stuff object, an object mean_width which gives the average width profile of the worm to be used for visualization, a function worm_maker which generates a worm as a vector, a function worm_drawer which rasterizes the worm, a standard_mask which gives us the proper shape of each panel of the final image, the function vertical_combine which combines images vertically, the function horizontal_combine which combines images horizontally, and output_file where the final output is saved, and draws a grid of normalized deviations of the worm along PCs. This also prints the cumulative proportion of variance explained from left to right.
	'''
	overall_list = []
	for j in range(0, len(my_PCA_stuff.components_))[0:20]:
		pose_list = []
		for i in range(-4, 5):
			pc_weights = np.zeros((len(my_PCA_stuff.components_)))
			pc_weights[j] = i
			my_x = pc_weights
			my_x = np.append(my_x, my_PCA_stuff.mean_scale)
			my_x = np.concatenate((my_x, np.array([0.5, 0.5])))
			my_x = np.append(my_x, 0)
			my_worm = worm_maker(my_PCA_stuff, my_x)
			raster_worm = worm_drawer(my_worm, my_PCA_stuff, my_x, canvas_shape = standard_mask.shape)
			pose_list.append(raster_worm)
		poses_together = vertical_combine(pose_list)
		overall_list.append(poses_together)
	all_together = horizontal_combine(overall_list)
	freeimage.write(all_together.astype('uint8'), output_file)

	so_far_var = []
	cumulative_var = 0	
	for i in range(0, 10):
		cumulative_var += (my_PCA_stuff.variances/sum(my_PCA_stuff.variances))[i]
		so_far_var.append(cumulative_var)
	print('Cumulative variances (left to right): ' + str(so_far_var))
	return

def training_fit(standard_mask, my_metadata_file, my_PCA, width_PCA, pcs_number = 30, n_points = 100, reverse_flag = False):
	'''
	Given a known spine of the worm, initialize a vector of parameters to input into our model.
	
	Returns the vector x0 which contains:
	pc_components
	scale_factor
	center_of_mass (normalized)
	long_axis_angle
	'''	
	# Initialize x0 and load stuff.
	x0 = np.zeros(pcs_number + 5 + 4)
	my_metadata = pickle.load(open(my_metadata_file, 'rb'))
	spine_tck = my_metadata['spine_tck']
	width_tck = my_metadata['width_tck']
	my_spine = zplib_interpolate.spline_interpolate(spine_tck, n_points)
	my_width = zplib_interpolate.spline_interpolate(width_tck, n_points)

	# Compute some parameters from the spine directly.
	center_of_mass = np.mean(my_spine, axis = 0)
	center_of_mass = np.divide(center_of_mass, standard_mask.shape)
	long_axis_angle = my_spine[-1, :] - my_spine[0, :]
	long_axis_angle = np.arctan2(long_axis_angle[1], long_axis_angle[0])
	spine_vectors = my_spine[1:] - my_spine[:-1]
	scale_factor = np.mean(np.array([np.linalg.norm(spine_vector) for spine_vector in spine_vectors]))

	# Fill in some simple parameters.
	x0[-3:-1] = center_of_mass
	x0[-1] = long_axis_angle 
	if reverse_flag:
		x0[-1] += np.pi
	x0[-4] = scale_factor

	# Project spine on to PCs.
	my_spine = normalize_worm(my_spine, final_angle = 0, final_scale = 1)
	linear_spine = np.ndarray.flatten(my_spine)
	my_components = deviation_projection(linear_spine, my_PCA)
	my_width_components = deviation_projection(my_width, width_PCA)
	
	# Fill in PCs and scale_factor, return x0.
	for i in range(0, pcs_number):
		x0[i] = my_components[i]		
	for i in range(0, 5):
		x0[pcs_number + i] = my_width_components[i]
	return x0

def fit_nspline(y_points, smoothing = None, order = None):
	'''
	Fit a non-parametric smoothing spline to a given set of x,y points.

	Parameters:
	points: array of n points x; shape=(n,)
	smoothing: smoothing factor: 0 requires perfect interpolation of the
		input points, at the cost of potentially high noise. Very large values
		will result in a low-order polynomial fit to the points. If None, an
		appropriate value based on the scale of the points will be selected.
	order: The desired order of the spline. If None, will be 1 if there are
		three or fewer input points, and otherwise 3.

	Returns a spline tuple (t,c,k) consisting of:
		t: the knots of the spline curve
		c: the x and y b-spline coefficients for each knot
		k: the order of the spline.

	Note: smoothing factor "s" is an upper bound on the sum of all the distances
	between the original x,y points and the matching points on the smoothed
	spline representation.
	'''
	y_points = np.asarray(y_points)
	l = len(y_points)
	if order is None:
		if l < 4:
			k = 1
		else:
			k = 3
	else:
		k = order

	# Choose input parameter values for the curve as the distances along the polyline:
	# This gives something close to the "natural parameterization" of the curve.
	# (i.e. a parametric curve with first-derivative close to unit magnitude: the curve
	# doesn't accelerate/decelerate, so points along the curve in the x,y plane
	# don't "bunch up" with evenly-spaced parameter values.)
	x_points = np.arange(y_points.shape[0])
	distances = np.asarray(x_points)
	distances = np.abs(distances[:-1] - distances[1:])
	distances = np.add.accumulate(distances)
	distances = np.concatenate([[0], distances])	

	if smoothing is None:
		smoothing = l * distances[-1] / 600.

	t, c, ier, msg = zplib_interpolate.splrep(distances, y_points, s=smoothing, k=k)
	if ier > 3:
		raise RuntimeError(msg)
	c[[0,-1]] = y_points[[0,-1]]
	return t,c,k


def initialize_fit(standard_mask, my_PCA, width_PCA, pcs_number = 20, n_points = 100, reverse_flag = False, verbose_mode = False, spine_mode = False):
	'''
	Given a rough mask of the worm, initialize a vector of parameters to input into our model.
	
	Returns the vector x0 which contains:
	pc_components
	scale_factor (relative to mean scale factor in my_PCA)
	center_of_mass (normalized)
	long_axis_angle
	'''	
	def initialize_outline(standard_mask, verbose_mode, reverse_flag = False):
		'''
		Given a mask of a worm, this function will guess at the outline of the worm, returning a spline fit to a pruned skeleton of the mask.
		'''
		def prune_skeleton(skeleton_graph, verbose_mode):
			'''
			Prune the graph of the skeleton so that only the single linear longest path remains.
			'''
			if verbose_mode:
				print('Pruning skeleton graph.')

			def farthest_node(my_graph, a_node, verbose_mode):
				'''
				Find the farthest node from a_node.
				'''
				reached_nodes = [a_node]
				distance_series = pd.Series(index = [str(a_node) for a_node in skeleton_graph.node_list])
				distance_series.loc[str(a_node)] = 0
				steps = 1
				current_circle = [a_node]
				next_circle = []
								
				while len(reached_nodes) < len(my_graph.node_list):
					if verbose_mode:		
						print('Reached nodes: ' + str(len(reached_nodes)))
					for current_node in current_circle:
						next_steps = my_graph.edges(current_node)
						if verbose_mode:
							print('Current circle')
							print(current_circle)
							print('next_steps')
							print(next_steps)
						for one_step in next_steps:
							other_node = [the_node for the_node in one_step if the_node != current_node][0]
							if other_node not in reached_nodes:
								distance_series.loc[str(other_node)] = steps 
								next_circle.append(other_node)
								reached_nodes.append(other_node)
					steps += 1
					current_circle = next_circle
					next_circle = []
				my_node = distance_series.argmax()
				my_node = (int(my_node[1:-1].split(',')[0]), int(my_node[1:-1].split(',')[1]))
				return my_node
			
			def find_minimal_path(my_graph, node_a, node_b, verbose_mode):
				'''
				Find the minimal path between node_a and node_b using my_graph.
				'''
				reached_nodes = [node_a]
				steps = 1
				current_circle = [[node_a]]
				next_circle = []
				got_to_b = False
				while not got_to_b:
					if verbose_mode:
						print(len(reached_nodes))		
						print([the_node for the_node in reached_nodes if the_node not in my_graph.node_list])		
					for current_path in current_circle:
						current_node = current_path[-1]
						next_steps = my_graph.edges(current_node)
						for one_step in next_steps:
							other_node = [the_node for the_node in one_step if the_node != current_node][0]
							if other_node == node_b:
								final_path = list(current_path)
								final_path.append(other_node)
								return (final_path, steps)
								
							elif other_node not in reached_nodes:
								next_path = list(current_path)
								next_path.append(other_node)
								next_circle.append(next_path)
								reached_nodes.append(other_node)
					steps += 1
					current_circle = next_circle
					next_circle = []	
				return
								
			one_end = farthest_node(skeleton_graph, skeleton_graph.node_list[0], verbose_mode = verbose_mode)
			if verbose_mode:
				print('First end is: ' + str(one_end))
			other_end = farthest_node(skeleton_graph, one_end, verbose_mode = verbose_mode)
			if verbose_mode:
				print('Second end is: ' + str(other_end))
				
			(my_path, path_length) = find_minimal_path(skeleton_graph, one_end, other_end, verbose_mode = verbose_mode)
			my_path = np.array(my_path)
			return my_path

		def skeletonize_mask(raster_worm, verbose_mode):
			'''
			Given a masked worm in raster format, return a skeletonized version of it.
			'''
			if verbose_mode:
				print('Skeletonizing mask.')
			zero_one_mask = np.zeros(raster_worm.shape)
			zero_one_mask[raster_worm > 0] = 1
			zero_one_mask = zplib_image_mask.get_largest_object(zero_one_mask)
			my_skeleton = skimage.morphology.skeletonize(zero_one_mask)
			skeleton_mask = np.zeros(raster_worm.shape).astype('uint8')
			skeleton_mask[my_skeleton] = -1
			return skeleton_mask
		
		def skeleton_to_graph(skeleton_mask, verbose_mode):
			'''
			Converts a skeleton to a graph, which consists of a dictionary with a list of nodes (tuples containing the coordinates of each node) in 'nodes' and a list of edges (lists of two tuples containing coordinates of the nodes connected by the edge; all edges have length 1).
			'''
			if verbose_mode:
				print('Converting skeleton to graph.')
			node_list = [tuple(a_point) for a_point in np.transpose(np.array(np.where(skeleton_mask > 0)))]
			edge_list = []
			for point_a in node_list:
				for point_b in node_list:
					distance_vector = np.array(point_a) - np.array(point_b)
					check_distance = np.max(np.abs(distance_vector))
					my_edge = sorted([point_a, point_b])
					if check_distance == 1:
						if my_edge not in edge_list:
							edge_list.append(my_edge)
			
			class a_graph():
				def __init__(self, node_list, edge_list):
					self.node_list = node_list
					self.edge_list = edge_list
					return
				def edges(self, a_node):
					return [an_edge for an_edge in edge_list if a_node in an_edge]
		
			my_graph = a_graph(node_list, edge_list)
			return my_graph
		
		messy_skeleton = skeletonize_mask(standard_mask, verbose_mode = verbose_mode)
		skeleton_graph = skeleton_to_graph(messy_skeleton, verbose_mode = verbose_mode)	
		pruned_graph = prune_skeleton(skeleton_graph, verbose_mode = verbose_mode)	
		if reverse_flag:
			pruned_graph = np.flipud(pruned_graph)
		spine_tck = zplib_interpolate.fit_spline(pruned_graph)
		return spine_tck
		
	def get_center_and_longaxis(raster_worm, my_spine):
		'''
		Given a masked worm in raster format and my_spine, find the normalized center of mass and the angle of the long axis.
		'''
		# Get normalized center of mass.
		standard_worm = np.array(np.where(raster_worm > 0))
		standard_center = np.mean(standard_worm, axis = 1)
		center_of_mass = np.divide(standard_center, raster_worm.shape)
		
		# Get long axis angle.
		long_axis = my_spine[-1, :] - my_spine[0, :]
		long_axis_angle = np.arctan2(long_axis[1], long_axis[0])
		visual_angle = np.array(np.arctan2(-long_axis[1], long_axis[0]))
		return (center_of_mass, long_axis_angle, visual_angle)

	# Initialize x0.
	x0 = np.zeros(pcs_number + 5 + 4)

	# Guess the spine and project it on to our PCA basis.
	if verbose_mode:
		print('Getting spine.')
	spine_tck = initialize_outline(standard_mask, reverse_flag = reverse_flag, verbose_mode = verbose_mode)
	my_spine = zplib_interpolate.spline_interpolate(spine_tck, n_points)

	# Fit a width profile to my mask.
	my_perpendiculars = zplib_geometry.find_polyline_perpendiculars(my_spine)
	distance_map = scipy.ndimage.morphology.distance_transform_edt(standard_mask)
	my_widths = np.zeros((my_spine.shape[0], 2))
	for i in range(0, len(my_perpendiculars)):		
		a_test_point = my_spine[i]
		b_test_point = my_spine[i]
		a_vector = my_perpendiculars[i]
		b_vector = -my_perpendiculars[i]
		found_edge_a = False
		found_edge_b = False
		a_side_distance = 0
		b_side_distance = 0
		while not(found_edge_a) or not(found_edge_b):
			a_side_distance += 1
			a_test_point = np.round(my_spine[i] + a_vector*a_side_distance)
			a_to_edge = distance_map[a_test_point[0], a_test_point[1]]
			if a_to_edge == 0 and not(found_edge_a):
				found_edge_a = True
				my_widths[i, 0] = a_side_distance
			b_side_distance += 1
			b_test_point = np.round(my_spine[i] + b_vector*b_side_distance)
			b_to_edge = distance_map[b_test_point[0], b_test_point[1]]
			if b_to_edge == 0 and not(found_edge_b):
				found_edge_b = True
				my_widths[i, 1] = b_side_distance
	my_widths = np.mean(my_widths, axis = 1)
	width_tck = fit_nspline(my_widths)
	if spine_mode:
		return (spine_tck, width_tck)

	# Get scale factor, based on length, for this worm.
	spine_vectors = my_spine[1:] - my_spine[:-1]
	scale_factor = np.mean(np.array([np.linalg.norm(spine_vector) for spine_vector in spine_vectors]))
	x0[-4] = scale_factor

	# Project spine on to PCs.
	normalized_spine = normalize_worm(my_spine, final_angle = 0, final_scale = 1)
	linear_spine = np.ndarray.flatten(normalized_spine)
	my_components = deviation_projection(linear_spine, my_PCA)
	for i in range(0, pcs_number):
		x0[i] = my_components[i]		

	# Project widths on to PCs.
	my_width_components = deviation_projection(my_widths, width_PCA)
	for i in range(0, 5):
		x0[pcs_number + i] = my_width_components[i]

	# Find center of mass and long axis angle.
	if verbose_mode:
		print('Getting center and long axis.')
	(center_of_mass, long_axis_angle, visual_angle) = get_center_and_longaxis(standard_mask, my_spine)	
	x0[-3:-1] = center_of_mass
	x0[-1] = long_axis_angle 
	return x0
	
def generate_grader(generate_specific_worm_from_parameters, mask_worm, mask_scorer, my_PCA, standard_mask_file, which_fixed = None, fixed_parameters = None):
	'''
	Generates a function that will grade a worm generated from parameters against standard_mask. It takes as input:
	
	generate_specific_worm_from_parameters: a function which will generate a set of outline points when given the mean worm as a set of outline points, the PCA object with the linear weights for each PC, a point around which to center the worm, and a set of weights to apply to the PC object.
	
	mask_worm: a function that will take a worm from a set of points and make a rasterized mask for comparison.
	
	mask_scorer: a function that will score the "goodness" of fit when given two masks of compatible size.
	
	my_PCA: an object with results from principal component analysis of the data.

	standard_mask_file: the file containing the mask that we want to fit our worm to.
	'''
	standard_mask = freeimage.read(standard_mask_file)	
	
	def my_grader(my_parameters):
		if which_fixed == None:
			my_fixed_parameters = np.array([False]*my_parameters.shape[0])
			my_which_fixed = np.array([False]*my_parameters.shape[0])
		else:
			my_fixed_parameters = fixed_parameters
			my_which_fixed = which_fixed		
		my_real_parameters = np.zeros(my_which_fixed.shape)
		my_real_parameters[my_which_fixed] = my_fixed_parameters[my_which_fixed]
		my_real_parameters[np.invert(my_which_fixed)] = my_parameters
		test_worm = generate_specific_worm_from_parameters(my_PCA, my_real_parameters)
		test_mask = mask_worm(test_worm, my_PCA, my_real_parameters, standard_mask.shape)	
		my_score = mask_scorer(test_mask, standard_mask)
		return my_score
	return my_grader	

def distance_pixels(test_mask, standard_mask):
	'''
	Grades test_mask against standard_mask by assigning points to each pixel based on distance from the edge of the standard_mask. For points outside the standard_mask, higher distance gives increasingly more loss. For points in the interior, a higher distance (more in center of worm) reduces the loss function.
	'''		
	# Make scoring mask.
	score_mask = scipy.ndimage.morphology.distance_transform_edt(standard_mask)
	score_mask = scipy.ndimage.morphology.distance_transform_edt(np.invert(standard_mask)) + -1*score_mask
	score_mask = score_mask.astype('int32')		
	
	# Now actually compute a score.
	my_score = np.sum(score_mask[test_mask > 0])
	return my_score


def main():
	return

if __name__ == "__main__":
	main()
