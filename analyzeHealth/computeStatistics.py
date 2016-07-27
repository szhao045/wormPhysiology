# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 19:18:04 2016

@author: Willie
"""

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.manifold
import multiprocessing
import random
import pickle
import os
import json

from zplib import pca as zplib_pca

import analyzeHealth.selectData as selectData

def divide_spans(complete_df):
	'''
	Divide individual variables into useful spans and return an array of transition times.
	'''
	span_variables = list(complete_df.key_measures)
	span_cutoffs = []
	for a_variable in span_variables:
		data_values = complete_df.mloc(measures = [a_variable])[:, 0, :]
		# Middle overall value.
		if a_variable in ['intensity_95', 'intensity_90', 'intensity_80', 'life_texture']:
			cutoff_value = np.ndarray.flatten(data_values)
			cutoff_value = cutoff_value[~np.isnan(cutoff_value)]
			cutoff_value = np.mean(cutoff_value)
			span_side = np.abs(data_values - cutoff_value)
			span_side = np.nanargmin(span_side, axis = 1)
			span_cutoffs.append(span_side)
		# Overall young adult value is 'healthy'.
		elif a_variable in ['bulk_movement']:
			young_adult = np.abs(complete_df.mloc(measures = ['egg_age'])[:, 0, :])
			young_adult = np.nanargmin(young_adult, axis = 1)
			cutoff_value = np.array([data_values[i, young_adult[i]] for i in range(0, young_adult.shape[0])])
			cutoff_value = np.mean(cutoff_value)/2
			span_side = np.abs(data_values - cutoff_value)
			span_side = np.nanargmin(span_side, axis = 1)
			span_cutoffs.append(span_side)
		# Point of own first maximum value.
		elif a_variable in ['total_size', 'cumulative_eggs', 'cumulative_area', 'adjusted_size']:
			span_side = np.nanargmax(data_values, axis = 1)
			span_cutoffs.append(span_side)
		# Raise an exception if I can't find the variable.
		else:
			raise BaseException('Can\'t find ' + a_variable + '.')
	return (span_variables, np.array(span_cutoffs))

def get_spans(adult_df, a_variable, method = 'young', fraction = 0.5, reverse_direction = False):
	'''
	Get healthspans as defined by young adult function for a_variable.
	'''
	data_values = adult_df.mloc(measures = [a_variable])[:, 0, :]
	if reverse_direction:
		data_values = data_values*-1
	if method == 'young':
		young_adult = np.abs(adult_df.mloc(measures = ['egg_age'])[:, 0, :])
		young_adult = np.nanargmin(young_adult, axis = 1)
		cutoff_value = np.array([data_values[i, young_adult[i]] for i in range(0, young_adult.shape[0])])
		cutoff_value = np.mean(cutoff_value)*fraction
	if method == 'overall_time':
		all_data = np.ndarray.flatten(data_values)
		all_data = all_data[~np.isnan(all_data)]
		cutoff_value = np.percentile(all_data, (1-fraction)*100)

	# Compute the time of crossing the health-gero threshold.
	span_side = data_values - cutoff_value
	health_span_length = np.nanargmin(np.abs(span_side), axis = 1)
	health_span_length = np.array([adult_df.ages[a_time]*24 for a_time in health_span_length])

	# Deal with the special cases of individuals that spend their entire lives in health or gerospan.
	span_list = [a_span[~np.isnan(a_span)] for a_span in span_side]
	only_health = np.array([(a_span > 0).all() for a_span in span_list])
	only_gero = np.array([(a_span < 0).all() for a_span in span_list])
	adultspans = selectData.get_adultspans(adult_df)
	health_span_length[only_health] = adultspans[only_health]
	health_span_length[only_gero] = 0
	return health_span_length

def differentiate(an_array, axis = 2):
	'''
	Differentiate an_array along axis and forward-fill a zero at the end so that the array maintains the same shape.
	'''
	if axis == 2:
		derivative_array = an_array[:, :, 1:] - an_array[:, :, :-1]
		derivative_array = np.concatenate((np.zeros((derivative_array.shape[0], derivative_array.shape[1], 1)), derivative_array), axis = 2)
	return derivative_array

def integrate(an_array, axis = 2):
	'''
	Integrate an_array along axis.
	'''
	integral_array = np.cumsum(an_array, axis = axis)
	return integral_array

def pincus2011_predict(complete_df, pincus_df):
	'''
	Try to get a big r^2 value with my variables.
	'''
	print('My correlations:')
	my_vars = list(complete_df.key_measures)
	my_vars.append('ghost_age')
	fivemean = complete_df.mloc(measures = my_vars)[:, :, 24:24+32+24]
	fivemean = np.median(fivemean, axis = 2)
	fivemean = fivemean[~np.isnan(fivemean).any(axis = 1)]
	for i in range(0, len(my_vars)):
		print('\t' + my_vars[i], scipy.stats.pearsonr(fivemean[:, i], fivemean[:, -1])[0]**2, scipy.stats.spearmanr(fivemean[:, i], fivemean[:, -1])[0]**2)

	print('\nPincus correlations:')
	my_vars = list(pincus_df.key_measures)
	my_vars.append('ghost_age')
	fivemean = pincus_df.mloc(measures = my_vars)[:, :, 24:24+32]
	fivemean = np.median(fivemean, axis = 2)
	fivemean = fivemean[~np.isnan(fivemean).any(axis = 1)]
	for i in range(0, len(my_vars)):
		print('\t' + my_vars[i], scipy.stats.pearsonr(fivemean[:, i], fivemean[:, -1])[0]**2, scipy.stats.spearmanr(fivemean[:, i], fivemean[:, -1])[0]**2)
	return

def health_to_mortality(complete_df, x_day_survival = 3):
	'''
	Convert health to mortality rate.
	'''
	health_data = complete_df.mloc(measures = ['health'])
	health_data = complete_df.display_variables(health_data, 'health')[0]
	death_data = complete_df.mloc(measures = ['ghost_age'])*(-1/24) < x_day_survival
	
	middle_healths = []
	mortality_rates = []
	for i in range(0, 23):
		health_cohort_mask = (i/2 < health_data) & (health_data < (i+1)/2)
		middle_health = (i/2 + (i+1)/2)/2
		total_in_cohort = np.sum(health_cohort_mask)
		total_mortality = np.sum(death_data[health_cohort_mask])
		mortality_rate = total_mortality/total_in_cohort		
		middle_healths.append(middle_health)
		mortality_rates.append(mortality_rate)
	return (middle_healths, mortality_rates)

	
def all_correlations(complete_df, my_variables = None):
	'''
	Check all n^2 correlations between my n variables.
	'''
	if my_variables == None:
		my_variables = list(complete_df.key_measures)
		my_variables.append('ghost_age')
		my_variables.append('age')
		
	my_array = []
	for a_var in my_variables:
		my_array.append(np.ndarray.flatten(complete_df.mloc(measures = [a_var])))
	my_array = np.array(my_array).transpose()
	my_array = my_array[~np.isnan(my_array).any(axis = 1)]
	
	my_array = pd.DataFrame(my_array, columns = my_variables)
	my_correlations = my_array.corr()
	return my_correlations**2

def check_dimensionality(complete_df, a_method, target_dimension, cohort_mode = False):
	'''
	Check the dimensionality of my data using a_method for target_dimension. If cohort_mode is True, then only check the dimensionality of a noise-smoothed/averaged version of my data. That should produce a lower number.
	'''
	# Select my data.	
	if cohort_mode:
		(my_cohorts, bin_lifes, my_bins, my_colors) = selectData.life_cohort_bins(complete_df, my_worms = complete_df.worms, bin_width_days = 2)
		my_data = []
		for a_cohort in my_cohorts:
			if len(a_cohort) > 0:
				my_data.append(np.mean(complete_df.mloc(measures = complete_df.key_measures)[a_cohort, :, :], axis = 0))
		my_data = np.array(my_data)
	else:
		my_data = complete_df.mloc(measures = complete_df.key_measures)

	# Cleverly flatten only two of the dimensions of a three-dimensional array.	
	my_data = np.swapaxes(my_data, 1, 2)
	my_data = np.reshape(my_data, (my_data.shape[0]*my_data.shape[1], my_data.shape[2]))
	my_data = my_data[~np.isnan(my_data).any(axis = 1)]	
	
	# Use isomap if it is the chosen method.
	if a_method == 'isomap':
		my_embedding = sklearn.manifold.Isomap(n_neighbors = 10, n_components = target_dimension)
		my_embedding.fit_transform(my_data)
		my_embedding.willie_error = my_embedding.reconstruction_error()
	elif a_method == 'modified_locally_linear_embedding':
		my_embedding = sklearn.manifold.LocallyLinearEmbedding(n_neighbors = 10, n_components = target_dimension, method = 'modified', eigen_solver = 'dense')
		my_embedding.fit_transform(my_data)
		my_embedding.willie_error = my_embedding.reconstruction_error_
	elif a_method == 'hessian_locally_linear_embedding':
		my_embedding = sklearn.manifold.LocallyLinearEmbedding(n_neighbors = (target_dimension*(target_dimension+3)//2) + 1, n_components = target_dimension, method = 'hessian', eigen_solver = 'dense')
		my_embedding.fit_transform(my_data)
		my_embedding.willie_error = my_embedding.reconstruction_error_
	elif a_method == 'ltsa_locally_linear_embedding':
		my_embedding = sklearn.manifold.LocallyLinearEmbedding(n_neighbors = 10, n_components = target_dimension, method = 'ltsa', eigen_solver = 'dense')
		my_embedding.fit_transform(my_data)
		my_embedding.willie_error = my_embedding.reconstruction_error_		
	elif a_method == 'locally_linear_embedding':
		my_embedding = sklearn.manifold.LocallyLinearEmbedding(n_neighbors = 10, n_components = target_dimension, method = 'standard', eigen_solver = 'dense')
		my_embedding.fit_transform(my_data)
		my_embedding.willie_error = my_embedding.reconstruction_error_		
	elif a_method == 'mds':
		my_embedding = sklearn.manifold.MDS(n_components = target_dimension, n_init = 1, max_iter = 100)
		my_embedding.fit_transform(my_data)
		my_embedding.willie_error = my_embedding.stress_

	return my_embedding

def quick_partial_pearson(independent_variable, dependent_variable, control_variable):
	'''
	Do a partial correlation between independent_variable and dependent_variable, controlled for control_variable.
	'''
	# Drop my null data.
	my_frame = np.array([independent_variable, dependent_variable, control_variable]).transpose()
	my_frame = my_frame[~np.isnan(my_frame).any(axis = 1)]
	(independent_variable, dependent_variable, control_variable) = [my_frame[:, i] for i in range(my_frame.shape[1])]

	# Calculate my partial correlation.
	pxy = scipy.stats.pearsonr(independent_variable, dependent_variable)[0]
	pxz = scipy.stats.pearsonr(independent_variable, control_variable)[0]
	pyz = scipy.stats.pearsonr(dependent_variable, control_variable)[0]
	partial_correlation = (pxy - pxz*pyz)/(np.sqrt((1-pxz**2)*(1-pyz**2)))
	return partial_correlation**2

def quick_pearson(independent_variable, dependent_variable, r_mode = False):
	'''
	Calculate a quick pearson r^2 while removing null data properly.
	'''
	my_frame = np.array([independent_variable, dependent_variable]).transpose().astype('float')
	my_frame = my_frame[~np.isnan(my_frame).any(axis = 1)]
	if r_mode:
		my_r = scipy.stats.pearsonr(my_frame[:, 0], my_frame[:, 1])[0]
		return my_r
	else:
		r_squared = scipy.stats.pearsonr(my_frame[:, 0], my_frame[:, 1])[0]**2
		return r_squared

def quick_spearman(independent_variable, dependent_variable):
	'''
	Calculate a quick spearman r^2 while removing null data properly.
	'''
	my_frame = np.array([independent_variable, dependent_variable]).transpose()
	my_frame = my_frame[~np.isnan(my_frame).any(axis = 1)]
	r_squared = scipy.stats.spearmanr(my_frame[:, 0], my_frame[:, 1])[0]**2
	return r_squared

def regression_check(complete_df, independent_variables, dependent_variable = 'ghost_age', my_times = None, cross_validate = True):
	'''
	A utility function to check the r^2 value of independent_variable against dependent_variable.
	'''
	if my_times == None:
		my_times = complete_df.times
	flat_dependent = np.ndarray.flatten(complete_df.mloc(measures = [dependent_variable], times = my_times))/24
	all_flats = []
	for a_var in independent_variables:
		flat_var = np.ndarray.flatten(complete_df.mloc(measures = [a_var], times = my_times))
		all_flats.append(flat_var)
	all_flats = np.array(all_flats).transpose()
	linear_r = quick_multiple_pearson(all_flats, flat_dependent)
	if cross_validate:
		cross_validated_r = cross_validate_pearson(all_flats, flat_dependent, k_fold = 10)
		return (independent_variables, linear_r, cross_validated_r)
	return (independent_variables, linear_r)

def multiple_regression_combine(complete_df, independent_variables, dependent_variable = 'ghost_age'):
	'''
	Use multiple regression to combine variables into a predictive score.
	'''
	flat_dependent = np.ndarray.flatten(complete_df.mloc(measures = [dependent_variable]))
	all_flats = []
	for a_var in independent_variables:
		flat_var = np.ndarray.flatten(complete_df.mloc(measures = [a_var]))
		all_flats.append(flat_var)
	all_flats = np.array(all_flats).transpose()
	(predicted_result, repeated_dependent, multiple_regression_weights, my_intercept) = multiple_regression(all_flats, flat_dependent)
	my_score = np.dot(multiple_regression_weights, np.transpose(all_flats)) + my_intercept
	my_score = np.reshape(my_score, (len(complete_df.worms), 1, len(complete_df.times)))
	return (my_score, multiple_regression_weights)

def cross_validate_pearson(independent_variables, dependent_variable, k_fold = 10):
	'''
	Obtain a cross-validated r^2 value.
	'''
	index_array = np.arange(independent_variables.shape[0])
	index_array = np.random.permutation(index_array)
	my_bins = np.array_split(index_array, k_fold)
	total_result = np.zeros(dependent_variable.shape)
	for i in range(0, len(my_bins)):
		training_mask = np.ones(dependent_variable.shape).astype('bool')
		training_mask[my_bins[i]] = False
		x_training_data = independent_variables[training_mask]
		y_training_data = dependent_variable[training_mask]
		
		(predicted_result, repeated_dependent, multiple_regression_weights, my_intercept) = multiple_regression(x_training_data, y_training_data)
		test_data = np.dot(multiple_regression_weights, np.transpose(independent_variables[~(training_mask)])) + my_intercept
		total_result[~(training_mask)] = test_data
				
	validated_r = quick_pearson(total_result, dependent_variable)
	return validated_r

def predict_life(complete_df, independent_variables = None, dependent_variable = 'ghost_age', my_times = None):
	'''
	Check regression r^2 values for independent_variables against dependent_variable. This assumes that dependent variable is a time in hours.
	'''
	if my_times == None:
		my_times = complete_df.times
	if independent_variables == None:
		independent_variables = complete_df.key_measures
	
	flat_dependent = np.ndarray.flatten(complete_df.mloc(measures = [dependent_variable], times = my_times))/24
	all_flats = []
	for a_var in independent_variables:
		flat_var = np.ndarray.flatten(complete_df.mloc(measures = [a_var], times = my_times))
		all_flats.append(flat_var)
	all_flats = np.array(all_flats).transpose()
	linear_r = quick_multiple_pearson(all_flats, flat_dependent)

	keep_indices = ~np.isnan(all_flats).any(axis = 1)
	selected_indices = np.zeros(np.count_nonzero(keep_indices)).astype('bool')
	selected_indices[:10000] = True
	selected_indices = np.random.permutation(selected_indices)
	all_flats = all_flats[keep_indices][selected_indices]
	flat_dependent = flat_dependent[keep_indices][selected_indices]
	
	# Do the coarse grid search. Hard-codes including 1/SVM_external_parameters['total_samples'] as a test value for gamma.
	my_workers = min(multiprocessing.cpu_count() - 1, 60)
	coarse_search_parameters = {'C': 2**np.arange(-10, 15).astype('float64'), 'gamma': np.hstack([2**np.arange(-15, 8).astype('float64'), 1/len(flat_dependent)]).astype('float64'), 'epsilon': 2**np.arange(-18, 7).astype('float64')}
	measure_svr = sklearn.svm.SVR()
	my_coarse_search = sklearn.grid_search.GridSearchCV(measure_svr, coarse_search_parameters, n_jobs = my_workers, verbose = 10, cv = 10)
	coarse_answer = my_coarse_search.fit(all_flats, flat_dependent)	
	
	measure_svr = sklearn.svm.SVR(C = coarse_answer['C'], gamma = coarse_answer['gamma'], epsilon = coarse_answer['epsilon'])
	measure_svr.fit(all_flats, flat_dependent)	
	svm_r = coarse_answer.best_score_
	return (linear_r, svm_r, coarse_answer)



def multiple_nonlinear_regression(complete_df, independent_variables, dependent_variable = 'ghost_age', method = 'svm'):
	'''
	Return an SVR trained to predict dependent_variable from independent_variables.
	'''
	independent_data  = np.array([np.ndarray.flatten(complete_df.mloc(measures = [independent_variable])) for independent_variable in independent_variables])
	dependent_data = np.ndarray.flatten(complete_df.mloc(measures = [dependent_variable]))
	together_data = np.vstack((independent_data, dependent_data)).transpose()
	together_data = together_data[~np.isnan(together_data).any(axis = 1)]
	independent_data  = together_data[:, :-1].copy()
	dependent_data = together_data[:, -1]
	measure_svr = sklearn.svm.SVR()
	measure_svr.fit(independent_data, dependent_data)	
	return (measure_svr, dependent_data, independent_data)

def quick_multiple_pearson(independent_variables, dependent_variable):
	'''
	Do multiple regression on an array of features (independent_variables) against dependent_variable and compute the pearson r^2.	
	'''
	(predicted_result, dependent_variable, multiple_regression_weights, my_intercept) = multiple_regression(independent_variables, dependent_variable)
	r_squared = scipy.stats.pearsonr(predicted_result, dependent_variable)[0]**2	
	return r_squared

def multiple_regression(independent_variables, dependent_variable):
	'''
	Do multiple regression on an array of features (independent_variables) against dependent_variable.
	'''
	my_regressor = np.concatenate([independent_variables, np.ones((independent_variables.shape[0], 1))], axis = 1)
	keep_indices = ~np.isnan(my_regressor).any(axis = 1)
	my_regressor = my_regressor[keep_indices]
	dependent_variable = dependent_variable[keep_indices]
	multiple_regression_weights = np.linalg.lstsq(my_regressor, dependent_variable)[0]
	my_intercept = multiple_regression_weights[-1]
	multiple_regression_weights = multiple_regression_weights[:-1]
	predicted_result = np.dot(multiple_regression_weights, my_regressor[:, :-1].transpose()) + my_intercept
	return (predicted_result, dependent_variable, multiple_regression_weights, my_intercept)

def trajectory_PCA(complete_df, components_PCA = None):
	'''
	Do PCA on my selected measurements!
	'''
	if components_PCA == None:
		components_PCA = complete_df.key_measures
	overall_frame = complete_df.mloc(measures = components_PCA)
	overall_frame = overall_frame.transpose([0, 2, 1])
	overall_frame = np.reshape(overall_frame, (-1, overall_frame.shape[2]))
	overall_frame = overall_frame[~np.isnan(overall_frame).any(axis = 1)]

	class a_PCA():
		def __init__(self, mean, pcs, norm_pcs, variances, positions, norm_positions, components_PCA):
			self.mean = mean
			self.pcs = pcs
			self.norm_pcs = norm_pcs
			self.variances = variances
			self.explained_variances = np.cumsum(self.variances/np.sum(self.variances))
			self.positions = positions
			self.norm_positions = norm_positions
			self.components_ = norm_pcs
			self.measures = components_PCA
			
	mean, pcs, norm_pcs, variances, positions, norm_positions = zplib_pca.pca(overall_frame)
	trajectory_PCA = a_PCA(mean, pcs, norm_pcs, variances, positions, norm_positions, components_PCA)
	return trajectory_PCA
	
def project_PCA(raw_data, trajectory_PCA):
	'''
	Projects an array of vectors on to our trajectory_PCA space.
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

	all_weights = np.zeros((raw_data.shape[0], len(trajectory_PCA.norm_pcs)))
	for i in range(0, raw_data.shape[0]):
		my_data = raw_data[i, :]
		my_component_weights = np.zeros(trajectory_PCA.norm_pcs.shape[0])
		for j in range(0, trajectory_PCA.norm_pcs.shape[0]):
			projection_len = scalar_project(my_data, trajectory_PCA.norm_pcs[j])
			component_length = np.linalg.norm(trajectory_PCA.norm_pcs[j])
			if component_length > 0:
				my_component_weights[j] = np.divide(projection_len, component_length)
			else: 
				my_component_weights[j] = 0
		all_weights[i, :] = my_component_weights.copy()
	return all_weights

def inflection_trajectory(x, y, set_endpoints = None):
	'''
	Compute the "inflection" of a trajectory (x, y) against the straight line that just goes from (x[0], y[0]) to (x[-1], y[-1]). Positive indicates that the worm was overall healthier than a linear decline.
	'''
	if set_endpoints == None:
		y = y - np.nanmin(y)
		y = y/np.nanmax(y)
		my_slope = (y[-1]-y[0])/(x[-1]-x[0])
#		my_slope = 1/(x[-1]-x[0])
	else:
		my_slope = (set_endpoints[-1]-set_endpoints[0])/(x[-1]-x[0])
	x_differences = x-x[0]
	straight_line_ys = my_slope*x_differences + y[0]
	inflection = np.mean(y - straight_line_ys)
	return inflection

def cross_validate_health(adult_df, independent_variables = ['autofluorescence', 'eggs', 'movement', 'size', 'texture'], dependent_variable = 'ghost_age'):
	'''
	'''
	worm_cohorts = np.array_split(adult_df.worms, 10)
	cohort_indices = [[adult_df.worm_indices[a_worm] for a_worm in a_cohort] for a_cohort in worm_cohorts]

	predicted_life = np.empty((len(adult_df.worms), len(adult_df.times)))

	i = 1	
	for a_cohort in worm_cohorts:
		print('cohort ', i)		
		i += 1
		training_group = [a_worm for a_worm in adult_df.worms if a_worm not in a_cohort]
	
		independent_data = np.array([np.ndarray.flatten(adult_df.mloc(worms = training_group, measures = [independent_variable])) for independent_variable in independent_variables])
		dependent_data = np.ndarray.flatten(adult_df.mloc(worms = training_group, measures = [dependent_variable]))
		together_data = np.vstack((independent_data, dependent_data)).transpose()
		together_data = together_data[~np.isnan(together_data).any(axis = 1)]
		independent_data  = together_data[:, :-1].copy()
		dependent_data = together_data[:, -1]
		texture_svr = sklearn.svm.SVR()
		texture_svr.fit(independent_data, dependent_data)	

		for a_worm in a_cohort:
			worm_data = np.array([np.ndarray.flatten(adult_df.mloc(worms = [a_worm], measures = [independent_variable])) for independent_variable in independent_variables]).transpose()	
			nonnan_values = ~(np.isnan(worm_data).any(axis = 1))
			worm_data = worm_data[nonnan_values]
		
			worm_prediction = texture_svr.predict(worm_data)
			predicted_life[adult_df.worm_indices[a_worm], nonnan_values] = worm_prediction

	print('Regular r^2 is ', quick_pearson(*adult_df.flat_data(['health', 'ghost_age'])))
	print('Cross-validated r^2 is ', quick_pearson(np.ndarray.flatten(predicted_life), adult_df.flat_data(['ghost_age'])[0]))
	return predicted_life

def cross_validated_health(adult_df, independent_variables = ['autofluorescence', 'eggs', 'movement', 'size', 'texture'], dependent_variable = 'ghost_age'):
	'''
	'''
	worm_cohorts = np.array_split(adult_df.worms, 10)
	cohort_indices = [[adult_df.worm_indices[a_worm] for a_worm in a_cohort] for a_cohort in worm_cohorts]

	predicted_life = np.empty((len(adult_df.worms), len(adult_df.times)))

	i = 1	
	for a_cohort in worm_cohorts:
		print('Up to cohort: ' + str(i))		
		i += 1
		training_group = [a_worm for a_worm in adult_df.worms if a_worm not in a_cohort]
	
		independent_data = np.array([np.ndarray.flatten(adult_df.mloc(worms = training_group, measures = [independent_variable])) for independent_variable in independent_variables])
		dependent_data = np.ndarray.flatten(adult_df.mloc(worms = training_group, measures = [dependent_variable]))
		together_data = np.vstack((independent_data, dependent_data)).transpose()
		together_data = together_data[~np.isnan(together_data).any(axis = 1)]
		independent_data  = together_data[:, :-1].copy()
		dependent_data = together_data[:, -1]
		texture_svr = sklearn.svm.SVR()
		texture_svr.fit(independent_data, dependent_data)	

		for a_worm in a_cohort:
			worm_data = np.array([np.ndarray.flatten(adult_df.mloc(worms = [a_worm], measures = [independent_variable])) for independent_variable in independent_variables]).transpose()	
			nonnan_values = ~(np.isnan(worm_data).any(axis = 1))
			worm_data = worm_data[nonnan_values]
		
			worm_prediction = texture_svr.predict(worm_data)
			predicted_life[adult_df.worm_indices[a_worm], nonnan_values] = worm_prediction

	print('Regular r^2 is ', quick_pearson(*adult_df.flat_data(['health', 'ghost_age'])))
	print('Cross-validated r^2 is ', quick_pearson(np.ndarray.flatten(predicted_life), adult_df.flat_data(['ghost_age'])[0]))
	return predicted_life

def svr_data(complete_df, independent_variables, dependent_variable = 'ghost_age'):
	'''	
	Compute predicted lifespan remaining for independent_variables using an SVM.
	'''
	# Get only non-null values to feed into my SVR.	
	(my_svm, dependent_data, independent_data) = multiple_nonlinear_regression(complete_df, independent_variables, dependent_variable)
	variable_data = complete_df.mloc(complete_df.worms, independent_variables)
	flat_variable_data = selectData.flatten_two_dimensions(variable_data, 1)
	nonnan_values = ~(np.isnan(flat_variable_data).any(axis = 1))
	flat_variable_data = flat_variable_data[nonnan_values]
	life_data = complete_df.mloc(complete_df.worms, [dependent_variable])[:, 0, :]
	life_data_shape = life_data.shape
	life_data = np.ndarray.flatten(life_data)
	flat_life_data_shape = life_data.shape

	# Predict remaining life using the SVR and then reshape my array back to its original shape.
	nonnan_svr_data = my_svm.predict(flat_variable_data)	
	svr_data = np.empty(flat_life_data_shape)
	svr_data[:] = np.nan
	svr_data[nonnan_values] = nonnan_svr_data
	svr_data = np.reshape(svr_data, life_data_shape)
	life_data = np.reshape(life_data, life_data_shape)
	return (variable_data, svr_data, life_data)
	
def one_d_geometries(complete_df, my_variable):
	'''
	Calculate some summary statistics about health declines.
	'''
	if my_variable in complete_df.measures:
		variable_data = complete_df.mloc(measures = [my_variable])[:, 0, :]
		(variable_data, my_unit, fancy_name) = complete_df.display_variables(variable_data, my_variable)
	else:
		variable_data = my_variable

	curve_data = {}
	nonnan_data = [variable_data[i, :][~np.isnan(variable_data[i, :])] for i in range(0, len(variable_data))]

	my_lifespans = selectData.get_adultspans(complete_df)/24
	my_starts = np.array([nonnan_data[i][0] for i in range(0, len(nonnan_data))])
	my_ends = np.array([nonnan_data[i][-1] for i in range(0, len(nonnan_data))])
	self_inflections = np.zeros(my_lifespans.shape)
	absolute_inflections = np.zeros(my_lifespans.shape)

	overall_start = np.mean(my_starts)
	overall_end = np.mean(my_ends)
	
	for i in range(0, len(nonnan_data)):
		worm_data = nonnan_data[i]
		worm_ages = np.array(complete_df.ages[:worm_data.shape[0]])
		self_inflection = inflection_trajectory(worm_ages, worm_data)
		absolute_inflection = inflection_trajectory(worm_ages, worm_data, set_endpoints = (overall_start, overall_end))
		self_inflections[i] = self_inflection
		absolute_inflections[i] = absolute_inflection
	
	curve_data['lifespan'] = my_lifespans
	curve_data['start'] = my_starts
	curve_data['end'] = my_ends
	curve_data['rate'] = (my_starts - my_ends)/my_lifespans
	curve_data['self_inflection'] = self_inflections
	curve_data['absolute_inflection'] = absolute_inflections
	return curve_data
	
def rate_consistency(complete_df, my_variable, number_of_chunks):
	'''
	For each worm in complete_df, split their lifespan/adultspan into number_of_chunks and return their average rate of change of my_variable over each chunk.
	'''
	variable_data = complete_df.mloc(measures = [my_variable])[:, 0, :]

	nonnan_data = [variable_data[i, :][~np.isnan(variable_data[i, :])] for i in range(0, len(variable_data))]
	nonnan_data = [np.array_split(worm_data, number_of_chunks) for worm_data in nonnan_data]	
	
	my_rates = pd.DataFrame([], index = complete_df.worms, columns = list(range(0, number_of_chunks)))

	my_lifespans = selectData.get_lifespans(complete_df)
	chunk_times = my_lifespans/number_of_chunks		
	
	for i in range(0, number_of_chunks):
		for j in range(0, len(nonnan_data)):
			my_start = nonnan_data[j][i][0]
			my_end = nonnan_data[j][i][-1]
			my_rates.iloc[j, i] = (my_start - my_end)/chunk_times[j]

	my_lifespans = selectData.get_lifespans(complete_df)
	my_rates.loc[:, 'lifespan'] = my_lifespans	
	return my_rates.astype('float')

def differences_in_aspects(complete_df, trajectory_PCA, group_a = None, group_b = None, my_measures = None):
	'''
	Determine whether worms differ in a number of ways.
	'''	
	def centroid_distance(population_a, population_b):
		'''
		Test whether the group of vectors population_a is different from the group of vectors population_b.
		'''
		a_centroid = np.mean(population_a, axis = 0)
		a_distances = np.linalg.norm(population_a - a_centroid, axis = 1)
		b_distances = np.linalg.norm(population_b - a_centroid, axis = 1)
		return scipy.stats.ttest_ind(a_distances, b_distances)

	# Make the two groups the halves of my population if they are not supplied.
	if group_a == None and group_b == None:
		raw_lifespans = selectData.get_lifespans(complete_df)	
		life_frame = pd.Series(raw_lifespans, index = complete_df.worms)
		middle_marker = life_frame.median()		
		group_a = list(life_frame[life_frame <= middle_marker].index)
		group_b = list(life_frame[life_frame > middle_marker].index)
	if my_measures == None:
		my_measures = complete_df.key_measures
	difference_frame = pd.DataFrame(index = ['start', 'end', 'journey'], columns = ['p', 't-statistic'])	
	
	# Compare starting points.
	a_starts = complete_df.mloc(group_a, my_measures)[:, :, 0]
	b_starts = complete_df.mloc(group_b, my_measures)[:, :, 0]
	start_stats = centroid_distance(a_starts, b_starts)
	difference_frame.loc['start', :] = (start_stats.pvalue, start_stats.statistic)
	
	# Compare ending points.
	a_ends = np.zeros((len(group_a), len(my_measures)))
	for i in range(0, len(my_measures)):
		a_measurement = my_measures[i]		
		my_data = complete_df.mloc(group_a, [a_measurement])[:, 0, :]
		for a_worm_index in range(0, my_data.shape[0]):
			worm_data = my_data[a_worm_index, :]
			last_point = worm_data[~np.isnan(worm_data)][-1]
			a_ends[a_worm_index, i] = last_point
	b_ends = np.zeros((len(group_b), len(my_measures)))
	for i in range(0, len(my_measures)):
		a_measurement = my_measures[i]		
		my_data = complete_df.mloc(group_b, [a_measurement])[:, 0, :]
		for a_worm_index in range(0, my_data.shape[0]):
			worm_data = my_data[a_worm_index, :]
			last_point = worm_data[~np.isnan(worm_data)][-1]
			b_ends[a_worm_index, i] = last_point
	end_stats = centroid_distance(a_ends, b_ends)
	difference_frame.loc['end', :] = (end_stats.pvalue, end_stats.statistic)

	# Compare displacements.
	a_displacements = a_ends - a_starts
	a_displacements = [np.linalg.norm(a_displacements[i, :]) for i in range(0, a_displacements.shape[0])]
	b_displacements = b_ends - b_starts
	b_displacements = [np.linalg.norm(b_displacements[i, :]) for i in range(0, b_displacements.shape[0])]
	journey_stats = scipy.stats.ttest_ind(a_displacements, b_displacements)
	difference_frame.loc['journey', :] = (journey_stats.pvalue, journey_stats.statistic)
	return difference_frame
	
def differentiate_trajectory(a_trajectory):
	'''
	Do a rough noisy differentiation of a_trajectory.
	'''
	differences = a_trajectory[1:, :] - a_trajectory[:-1, :]
	return differences
	
def main():
	return

if __name__ == "__main__":
	main()
