#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:38:09 2021

@author: msmuhammad
title: cNMF tutorial
"""

#imprting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib import gridspec

from cNMF.cnmf import cNMF


# download your example data
# ! wget O ./example_simulated_data.tar.gz https://storage.googleapis.com/sabeti-public/dkotliar/cNMF/example_data_20191024.tar.gz
# extract
# ! tar -zxvf ./example_simulated_data.tar.gz && rm ./example_simulated_data.tar.gz



###########################################
# cNMF run parameters

# this parameter should be set to a larger value for real data. 
numiter = 20

# Set this to a larger value and use the parallel code cells to try out parallelization
numworkers = 1

## Number of over-dispersed genes to use for running the factorizations
numhvgenes = 1500

K = ' '.join([str(i) for i in range(5,10)])

# Results will be saved to [output_directory]/[run_name] which in this example is simulated_example_data/example_cNMF

output_directory = './simulated_example_data'
run_name = 'example_cNMF'

countfn = './simulated_example_data/filtered_counts.txt'
seed = 14




#######################################
# preparing the normalized count matrix of high var gene and the cNMF\
    # parameters file (assume no parallelization)

# this is to normalize the count matrix and choose the highewr 2000 \
    # most over-dispersed genes to run the cNMF on

# It indicates that it will run 5 NMF iterations each for \
    # K=4, 5, 6, 7, and 8. With one worker

prepare_cmd = 'python cnmf.py prepare --output-dir %s --name %s -c %s -k %s --n-iter %d --total-workers %d --seed %d --numgenes %d' % (output_directory, run_name, countfn, K, numiter, numworkers, seed, numhvgenes)
# print('Prepare command:\n%s' % prepare_cmd)
# ! {prepare_cmd}
#  write the next command line in the terminal instead of the one baove
#  python cnmf.py prepare --output-dir ./simulated_example_data --name example_cNMF -c ./simulated_example_data/filtered_counts.txt -k 5 6 7 8 9 --n-iter 100 --total-workers 1 --seed 14 --numgenes 1500

factorize_cmd = 'python cnmf.py factorize --output-dir %s --name %s --worker-index 0' % (output_directory, run_name)
# print('Factorize command for worker 0:\n%s' % factorize_cmd)
# ! {factorize_cmd}



# Combine the replicate spectra into merged spectra files
cmd = 'python cnmf.py combine --output-dir %s --name %s' % (output_directory, run_name)
# print('Combine command:\n%s' % cmd)
# ! {cmd}



# Plot the trade-off between error and stability as a function of K to guide selection of K
plot_K_selection_cmd = 'python cnmf.py k_selection_plot --output-dir %s --name %s' % (output_directory, run_name)
# print('Plot K tradeoff command:\n%s' % plot_K_selection_cmd)
# !{plot_K_selection_cmd}


Image(filename = "./simulated_example_data/example_cNMF/example_cNMF.k_selection.png", width=1000, height=1000)



# We proceed to obtain the consensus matrix factorization estimates
selected_K = 7 



## This is the command you would run from the command line to obtain the consensus estimate with no filtering
## and to save a diagnostic plot as a PDF
consensus_cmd = 'python cnmf.py consensus --output-dir %s --name %s --local-density-threshold %.2f --components %d --show-clustering' % (output_directory, run_name, 2.00, selected_K)
# print('Consensus command for K=%d:\n%s' % (selected_K, consensus_cmd))
# !{consensus_cmd}

from IPython.display import Image
# Image(filename = "./simulated_example_data/example_cNMF/example_cNMF.clustering.k_%d.dt_2_00.png" % selected_K, width=1000, height=1000)

density_threshold = 0.10
density_threshold_str = '0_10'



consensus_cmd = 'python cnmf.py consensus --output-dir %s --name %s --local-density-threshold %.2f --components %d --show-clustering' % (output_directory, run_name,density_threshold, selected_K)
# print('Command: %s' % consensus_cmd)
# ! {consensus_cmd}

from IPython.display import Image
Image(filename = "./simulated_example_data/example_cNMF/example_cNMF.clustering.k_%d.dt_%s.png" % (selected_K, density_threshold_str), width=1000, height=1000)


