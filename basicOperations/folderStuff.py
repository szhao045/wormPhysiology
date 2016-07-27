# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 05:47:24 2015

@author: Willie
"""
import os
import sys
import shutil
import pathlib
import subprocess
import json
import pandas as pd

def push_code(source_folder = '', destination = 'wzhang@heavenly.wucon.wustl.edu:/mnt/bulkdata/wzhang', my_password = 'wustlzplab'):
	'''
	Update all my code at destination using the source at source, using the password my_password for the destination password.
	
	NOTE: THIS ASSUMES THAT THE DESTINATION FOLDER ALREADY HAS THE NECESSARY SUBDIRECTORY STRUCTURE FOR THE .PY SCRIPT FILES.
	'''
	destination_split = destination.split(':')
	destination_machine = destination_split[0].split('@')[-1]	
	destination_folder = destination_split[1]
	
	if source_folder == '':
		file_dir = os.path.dirname(__file__)
		source_folder = str(pathlib.Path(file_dir).parent)
		confirmation_string = 'Are you sure you want to push all .py files from ' + source_folder + ' on this machine to ' + destination_folder + ' on ' + destination_machine + '? (Y/N)' + '\n: '
		my_answer = input(confirmation_string)
		if my_answer.lower() != 'y':
			raise BaseException('\tFile pushing canceled')
	
	def push_file(file_path, destination, my_password, pscp_path = r'"C:\Users\Willie\Desktop\PortableApps\PuTTYPortable\App\putty\PSCP.EXE"', print_commands = False):
		'''
		Push a single file using PSCP from PuTTY.
		'''
		my_command = pscp_path
		my_command += ' -pw ' + my_password
		my_command += ' "' + file_path + '"'
		my_command += ' "' + destination + '"'
		if print_commands:
			print('\t', my_command)
		my_output = subprocess.check_output(my_command)
		if print_commands:
			print('\t', my_output)
		return

	source_path = pathlib.Path(source_folder)
	destination_path = pathlib.PurePosixPath(destination_folder)
	my_python_files = list(source_path.glob('**/*.py'))
	source_files = [str(a_path) for a_path in source_path.glob('**/*.py')]
	destination_folders = [str((destination_path / pathlib.PurePosixPath(a_file.relative_to(source_path))).parent) for a_file in my_python_files]
	destination_folders = [destination.replace(destination_folder, final_destination_folder) for final_destination_folder in destination_folders]

	for i in range(0, len(source_files)):
		print('\tPushing file ' + str(i+1) + '/' + str(len(source_files)) + ': ' + '/'.join(source_files[i].split(os.path.sep)[-2:]))
		push_file(source_files[i], destination_folders[i], my_password)
	print('Pushing completed!')
	return

def copy_exp_metadata(data_dir, human_dir, my_prefix = ''):
	'''
	Copy metadata to my validated data directory.
	'''
	for a_subdir in os.listdir(data_dir):
		if os.path.isdir(data_dir + os.path.sep + a_subdir):
			for a_file in os.listdir(data_dir + os.path.sep + a_subdir):
				if a_file == 'position_metadata_extended.json':
					if os.path.isdir(human_dir + os.path.sep + my_prefix + a_subdir):
						meta_src = data_dir + os.path.sep + a_subdir + os.path.sep + a_file
						meta_dst = human_dir + os.path.sep + my_prefix + a_subdir + os.path.sep + a_file
						if not os.path.isfile(meta_dst):
							print('Copying ' + meta_src + ' to ' + meta_dst + '.')
							shutil.copyfile(meta_src, meta_dst)
	return

def copy_big_metadata(directory_bolus, destination_folder):
	'''
	Copy overall experiment metadata to individual folders.
	'''
	for data_dir in directory_bolus.data_directories:
		old_meta_name = data_dir + os.path.sep + 'experiment_metadata.json'
		new_meta_name = destination_folder + os.path.sep + data_dir.split(os.path.sep)[-1].replace(' Run ', ' ') + '.json'
		shutil.copyfile(old_meta_name, new_meta_name)
	return

def copy_exp_masks(work_dir, human_dir, my_prefix = ''):
	'''
	Copy masks and bright-field images to my validated data directory.
	'''
	for a_subdir in sorted(list(os.listdir(work_dir))):
		if os.path.isdir(work_dir + os.path.sep + a_subdir):
			if not os.path.exists(human_dir + os.path.sep + my_prefix + a_subdir):
				for a_file in os.listdir(work_dir + os.path.sep + a_subdir):
					if a_file.split(' ')[-1] == 'mask.png':
						ensure_folder(human_dir + os.path.sep + my_prefix + a_subdir)
						mask_src = work_dir + os.path.sep + a_subdir + os.path.sep + a_file
						mask_dst = human_dir + os.path.sep + my_prefix + a_subdir + os.path.sep + a_file.replace('mask.png', 'hmask.png')
						bf_src = mask_src.replace('mask.png', 'bf.png')
						bf_dst = mask_dst.replace('hmask.png', 'bf.png')
						shutil.copyfile(mask_src, mask_dst)
						shutil.copyfile(bf_src, bf_dst)
	return

def ensure_folder(folder_path):
	'''
	Make sure a folder exists and make it if necessary.
	'''
	try:
		os.stat(folder_path)
	except: 
		os.mkdir(folder_path)
	return

def linux_path(a_path):
	'''
	Converts a windows path on the internal WUSTL network into a mounted linux path as seen by heavenly.
	'''
	if a_path == None:
		return a_path
	new_path = a_path.replace(r'\\heavenly.wucon.wustl.edu', r'/mnt/bulkdata')
	new_path = new_path.replace(r'\\zpl-scope.wucon.wustl.edu\scopearray', r'/mnt/scopearray')
	new_path = new_path.replace('\\', '/')
	return new_path

def move_bright_fields(list_of_images, directory_bolus, viewing_directory):
	'''
	Move the list of indicated bright_fields returned by selectData.rank_worms for convenient viewing.
	'''
	for a_string in list_of_images:
		split_string = a_string.split(' ')
		the_worm = ' '.join(split_string[:-1])
		the_time = split_string[-1]
		source_image = directory_bolus.working_directory + os.path.sep + the_worm + os.path.sep + the_time +  ' ' + 'bf.png'
		destination_image = viewing_directory + os.path.sep + a_string +  ' ' + 'bf.png'
		shutil.copyfile(source_image, destination_image)
	return

class DirectoryBolus():
	'''
	A class to store all my directory locations in.
	'''
	def __init__(self, working_directory, human_directory, data_directories, extra_directories, experiment_directories, annotation_directories, ready = 0, done = 0):
		self.ready = ready
		self.done = done
		self.working_directory = working_directory
		self.human_directory = human_directory
		self.data_directories = data_directories
		self.extra_directories = extra_directories
		self.experiment_directories = experiment_directories
		self.annotation_directories = annotation_directories
		self.platform = sys.platform
		if self.platform == 'linux':
			self.linux_paths()
		self.windows_health = r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016 spe-9 Measured Health'
	
	def counts(self):
		'''
		Check counts of all directory lists.
		'''
		lists_list = [self.data_directories, self.extra_directories, self.experiment_directories, self.annotation_directories]
		string_list = ['Data Directories', 'Extra Directories', 'Experiment Directories', 'Annotation Directories']
		for i in range(0, len(lists_list)):
			print(string_list[i] + ':' + str(len(lists_list[i])))
		return		
		
	def linux_paths(self):
		'''
		Convert all paths to linux paths.
		'''
		self.working_directory = linux_path(self.working_directory)
		self.human_directory = linux_path(self.human_directory)
		self.data_directories = [linux_path(data_directory) for data_directory in self.data_directories]
		new_extra_directories = []
		for extra_dictionary in self.extra_directories:
			if extra_dictionary == None:
				new_extra_directories.append(None)
			else:
				new_extra_directories.append({a_key: linux_path(extra_dictionary[a_key]) for a_key in extra_dictionary.keys()})
		self.extra_directories = list(new_extra_directories)
		self.annotation_directories = [linux_path(annotation_directory) for annotation_directory in self.annotation_directories]		
		return
	
	def super_checker_arguments(self, an_index):
		'''
		Return arguments for initializing a HumanCheckpoints object for the experiment with index an_index.
		'''
		# Figure out the metadata.
		if self.extra_directories[an_index] != None:	
			my_arguments = (self.annotation_directories[an_index], self.extra_directories[an_index])
		else:
			my_arguments = (self.data_directories[an_index], {})
		return my_arguments
	

def main():
	return

if __name__ == "__main__":
	main()
