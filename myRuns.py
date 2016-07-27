# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:55 2015

@author: Willie

THE FOLLOWING IS MY HIERARCHY OF IMPORTS:
	myRuns CAN IMPORT ANYTHING. ANYTHING!!!
	mainFigures can import anything below this.
	supplementFigures can import anything below this.
	plotFigures can import anything below this.
	characterizeTrajectories can import anything below this.
	computeStatistics can import anything below this.
	selectData can import anything below this.
	organizeData can import anything below this.
	extractFeatures can import anything below this.
	backgroundSubtraction, edgeMorphology, shapeModelFitting, and textureClassification can import anything below this.
	imageOperations can import anything below this.
	folderStuff can import anything below this.
	Python libraries.
"""

import sys
import os.path
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
os.chdir(file_dir)

import numpy as np
import json
import pathlib
import os
import pickle
import shutil
import pandas as pd	
import sklearn	
import random
import scipy.cluster
import pathlib
import time
import scipy.stats
import scipy.interpolate
import freeimage
import concurrent.futures
import multiprocessing
import time 
import matplotlib.pyplot as plt

import zplib.image.resample as zplib_image_resample
import zplib.image.mask as zplib_image_mask

import basicOperations.folderStuff as folderStuff
import basicOperations.imageOperations as imageOperations
import wormFinding.backgroundSubtraction as backgroundSubtraction	
import wormFinding.shapeModelFitting as shapeModelFitting
import wormFinding.textureClassification as textureClassification
import wormFinding.edgeMorphology as edgeMorphology
import measurePhysiology.extractFeatures as extractFeatures
import measurePhysiology.organizeData as organizeData
import analyzeHealth.selectData as selectData
import analyzeHealth.computeStatistics as computeStatistics
import analyzeHealth.characterizeTrajectories as characterizeTrajectories
import graphingFigures.plotFigures as plotFigures
import graphingFigures.cannedFigures as cannedFigures
import graphingFigures.supplementFigures as supplementFigures
import graphingFigures.mainFigures as mainFigures
import graphingFigures.finalSupplement as finalSupplement

print('Getting started!', flush = True)	
# Default to my_mode 0.
my_mode = 0
if len(sys.argv) > 1:
	my_mode = int(sys.argv[1])

# Locations of all the files I need.
save_directory = r'C:\Users\Willie\Desktop\save_dir'
working_directory = r'\\heavenly.wucon.wustl.edu\wzhang\work_dir'
human_directory = r'\\heavenly.wucon.wustl.edu\wzhang\human_dir'
data_directories = [
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.16 spe-9 Run 9',      #0
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.20 spe-9 Run 10A',    #1
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.20 spe-9 Run 10B',    #2
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11A',    #3
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11B',    #4
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11C',    #5
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11D',    #6
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.29 spe-9 Run 12A',    #7
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.29 spe-9 Run 12B',    #8
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13A',    #9
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13B',    #10
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13C',    #11
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.14 spe-9 Run 14',     #12
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.25 spe-9 Run 15A',    #13
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.25 spe-9 Run 15B',    #14
	r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.31 spe-9 Run 16',      #15
	r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17A', #16
	r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17B', #17
	r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17C', #18
	r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17D', #19
	r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17E', #20
]
extra_directories = [
	{'W': r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.02.16 spe-9 Run 9'},      #0
	None,                                                                                     #1
	None,                                                                                     #2
	None,                                                                                     #3
	None,                                                                                     #4
	None,                                                                                     #5
	None,                                                                                     #6
	{'W': r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.02.29 spe-9 Run 12A'},    #7
	{'W': r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.02.29 spe-9 Run 12B'},    #8
	None,                                                                                     #9
	{'W': r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.03.04 spe-9 Run 13B'},    #10
	{'W': r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.03.04 spe-9 Run 13C'},    #11
	None,                                                                                     #12
	None,                                                                                     #13
	None,                                                                                     #14
	None,                                                                                     #15
	None,                                                                                     #16
	None,                                                                                     #17
	None,                                                                                     #18
	None,                                                                                     #19
	None,                                                                                     #20
]
experiment_directories = [
	'W',                                                                                      #0
	None,                                                                                     #1
	None,                                                                                     #2
	None,                                                                                     #3
	None,                                                                                     #4
	None,                                                                                     #5
	None,                                                                                     #6
	'W',                                                                                      #7
	'W',                                                                                      #8
	None,                                                                                     #9
	'W',                                                                                      #10
	'W',                                                                                      #11
	None,                                                                                     #12
	None,                                                                                     #13
	None,                                                                                     #14	
	None,                                                                                     #15		
	None,                                                                                     #16
	None,                                                                                     #17
	None,                                                                                     #18
	None,                                                                                     #19
	None,                                                                                     #20
]
annotation_directories = [
	r'\\zpl-scope.wucon.wustl.edu\scopearray\Sinha_Drew\20160216_spe9Acquisition',            #0
	None,                                                                                     #1
	None,                                                                                     #2
	None,                                                                                     #3
	None,                                                                                     #4
	None,                                                                                     #5
	None,                                                                                     #6
	r'\\zpl-scope.wucon.wustl.edu\scopearray\Sinha_Drew\20160229_spe9Acquisition_DevVarA',    #7
	r'\\zpl-scope.wucon.wustl.edu\scopearray\Sinha_Drew\20160229_spe9Acquisition_DevVarB',    #8
	None,                                                                                     #9
	r'\\zpl-scope.wucon.wustl.edu\scopearray\Sinha_Drew\20160304_spe9Acquisition_DevVarB',    #10
	r'\\zpl-scope.wucon.wustl.edu\scopearray\Sinha_Drew\20160304_spe9Acquisition_DevVarC',    #11
	None,                                                                                     #12
	None,                                                                                     #13
	None,                                                                                     #14
	None,                                                                                     #15
	None,                                                                                     #16
	None,                                                                                     #17
	None,                                                                                     #18
	None,                                                                                     #19
	None,                                                                                     #20
]
directory_bolus = folderStuff.DirectoryBolus(working_directory, human_directory, data_directories, extra_directories, experiment_directories, annotation_directories, done = 21, ready = 21)	
if sys.platform == 'linux':
	human_directory = folderStuff.linux_path(human_directory)
	working_directory = folderStuff.linux_path(working_directory)

# Initialize a checker for manual image annotation.
# Use the following if you have issues with file permissions: sudo chmod -R g+w /mnt/scopearray
if my_mode == 1:
	checker = organizeData.HumanCheckpoints(data_directories[-1])

# Do the image analysis on heavenly/debug it.
if my_mode == 2:
	organizeData.measure_experiments(
		# Directory information.
		directory_bolus,

		# Parameters for broken down runs for debugging.
		parallel = True, 
		only_worm = None, 
		only_time = None,
#		parallel = False, 
#		only_worm = directory_bolus.data_directories[15] + os.path.sep + '132', 
#		only_time = '2016-02-23t0041',

		# Metadata information.
		refresh_metadata = True
	)

# Do the analysis.
if my_mode == 3: 
	adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True})	
	