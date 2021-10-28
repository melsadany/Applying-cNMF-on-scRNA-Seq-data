#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:10:41 2021

@author: msmuhammad

Title: cNMF_paper
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import sys
import glob, os

import cnmf
from cnmf import save_df_to_npz, get_highvar_genes


sys.path.append('/wdata/msmuhammad/projects/rotation/rotation-1-scrna-seq/code/cNMF')
# I couldn't import this function
import plotting.py
from plotting import make_genestats_plot

counts_fn = 'GSE102827/GSE102827_merged_all_raw.csv'




txt = open(counts_fn).readlines()

df = pd.DataFrame(columns= [i for i in range(65540)], index= [i for i in range(25189)])
for i, ln in enumerate(txt):
    row_items = ln.split(",")
    df.loc[i,] = row_items
    
counts = df
counts.index = df.loc[:,0]
counts.columns = df.iloc[0,]
n = 1
#  drop the last row, it's empty
counts = counts[:-1]
counts = counts.iloc[1:, 1:]


#  I didn't save the df to npz, because I don't get the rationale behind it!
save_df_to_npz(counts, 'GSE102827/GSE102827_merged_all_raw.npz')


celltype = pd.read_csv('GSE102827/GSE102827_cell_type_assignments.csv', index_col=0)
celltype['maintype'].value_counts()

# Dropping non-neuronal cells 
# the counts.columns in the line below is changed from the paper code. I changed it because 
# I don't think they mean the index in this case!
neuron_ind = celltype.loc[counts.columns, 'maintype'].isin(['Excitatory', 'Interneurons'])
non_neuron_cells = neuron_ind.index[~neuron_ind]
counts_neuronal = counts
counts_neuronal.drop(non_neuron_cells, axis=1, inplace=True)

counts_neuronal.to_csv('GSE102827/GSE102827_neuronal_cells.csv')


counts_neuronal_2 = pd.read_csv('GSE102827/GSE102827_neuronal_cells.csv', sep=',', index_col=0).T

# Dropping low count genes and cells

(fig,ax) = plt.subplots(1,1, figsize=(2,2), dpi=300)
countcutoff=3.0

counts_per_cell = counts_neuronal_2.sum(axis=1)
counts_neuronal_2.drop(counts_per_cell.index[counts_per_cell==0], axis=0, inplace=True)
counts_per_cell = counts_per_cell.loc[counts_per_cell>0]


ax.hist(counts_per_cell.apply(np.log10), bins=90)
_ = ax.set_title('Log10 Counts Per Cell')
lims = ax.get_ylim()
ax.vlines(x=countcutoff, ymin=0, ymax=lims[1], linestyle='--', label='minimum threshold')
_ = ax.set_ylim(lims)



counts_neuronal_2.to_csv('GSE102827/GSE102827_neuronal_cells_T_low_counts_dropped.csv')







TPM = counts_neuronal_2.div(counts_neuronal_2.sum(axis=1), axis=0) * (10**6)
TPM.to_csv('GSE102827/GSE102827_neuronal_cells_T_low_counts_dropped_TPM.csv')

nnzthresh = counts_neuronal_2.shape[0] / 500

numnonzero = (counts_neuronal_2>0).sum(axis=0)
print((numnonzero>nnzthresh).value_counts())
ind = numnonzero<200
(fig,ax) = plt.subplots(1,1, figsize=(2,2), dpi=300)
_ = ax.hist(numnonzero.loc[ind], bins=100)
(_,ymax) = ax.get_ylim()
ax.vlines(x=nnzthresh, ymin=0, ymax=ymax, linestyle='--')
ax.set_xlabel('# Samples With Non-zero Count')
ax.set_ylabel('# Genes')




genestodrop = numnonzero.index[(numnonzero<=nnzthresh)]
counts_neuronal_2.drop(genestodrop, axis=1, inplace=True)
TPM.drop(genestodrop, axis=1, inplace=True)


cellstodrop = counts_per_cell.index[counts_per_cell<(10**countcutoff)]
counts_neuronal_2.drop(cellstodrop, axis=0, inplace=True)
TPM.drop(cellstodrop, axis=0, inplace=True)

counts_neuronal_2.to_csv('GSE102827/GSE102827_neuronal_cells_T_low_counts_dropped_samples_nonzero.csv')
TPM.to_csv('GSE102827/GSE102827_neuronal_cells_T_low_counts_dropped_samples_nonzero_TPM.csv')


save_df_to_npz(TPM, 'GSE102827/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM.npz')
save_df_to_npz(counts_neuronal_2, 'GSE102827/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_counts.npz')



(genestats, gene_fano_parameters) = get_highvar_genes(TPM, numgenes=2000, minimal_mean=0)
# I couldn't reproduce the graphs using this function because I wasn't able to import it 
# I couldn't import the function because it doesn't exist in the module files!!!!
axes = make_genestats_plot(genestats, highvarcol='high_var')


genestats_fn = 'GSE102827/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM_genestats.txt'
highvar_genes_fn = 'GSE102827/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM_filter_2000hvgs.txt'
genestats.to_csv(genestats_fn, sep='\t')
highvar_genes = genestats.index[genestats['high_var']]
open(highvar_genes_fn, 'w').write('\n'.join(highvar_genes))




#################################################################################################
# the cNMF for Hravatin dataset

import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cnmf import load_df_from_npz, save_df_to_npz, cNMF

# same issue again!!!!
from plotting import plot_comparison, labeled_heatmap



from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib as mpl


base_directory = '/wdata/msmuhammad/projects/rotation/rotation-1-scrna-seq/code/cNMF'
project_name = 'GSE102827_cNMF'
countfn = os.path.join(base_directory, 'GSE102827/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_counts.npz')
# K = ' '.join([str(k) for k in range(15,30)])
K = ' '.join([str(15), str(30)])
masterseed = 31
numboot=10
nworkers=1
highvargenesfn = os.path.join(base_directory,'GSE102827/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM_filter_2000hvgs.txt')


cmd = 'python cnmf.py prepare --output-dir %s --name %s -c %s -k %s --n-iter %d --total-workers %d --seed %d --genes-file %s' % (base_directory, project_name, countfn, K, numboot, nworkers, masterseed, highvargenesfn)
!{cmd}


# I used only one worker and 2 values for K with 10 iterations for each 

worker_index = ' '.join([str(x) for x in range(nworkers)]) 

# factorize_cmd = 'python cnmf.py factorize --output-dir %s --name %s --counts %s -k %s --n-iter %d --total-workers %d --worker-index {} ::: %s' % (base_directory, project_name, countfn, K, numboot, nworkers, worker_index) 
factorize_cmd = 'python cnmf.py factorize --output-dir %s --name %s --counts %s -k %s --n-iter %d --total-workers %d' % (base_directory, project_name, countfn, K, numboot, nworkers) 
!{factorize_cmd}


cmd = 'python cnmf.py combine --output-dir %s --name %s' % (base_directory, project_name) 
print(cmd) 
!{cmd}




# I didn't use the output of my code and used the uploaded files from the paper
# I skipped the cNMF running, plotting to choose K, and the merging step[]


from cnmf import load_df_from_npz, save_df_to_npz, cNMF


# In the next line, I depended on the merged K 20 file from the paper
#I downloaded the file and added it to my cNMF_tmp directory
cnmf_obj = cNMF(output_dir=base_directory, name=project_name)
cnmf_obj._initialize_dirs()
cnmf_obj.consensus(k=20, density_threshold_str='2.0', local_neighborhood_size = 0.30,show_clustering = True, close_clustergram_fig=True)


#####################################################################
# filtering and setting a threshold 


cnmf_obj.consensus(k=20, density_threshold_str='0.08', local_neighborhood_size = 0.30,show_clustering = True, close_clustergram_fig=True)
###################################################################
# plotting

## Output a slightly improved clustergram figure for the manuscript using the exact same procedure

from scipy.spatial.distance import squareform
from fastcluster import linkage
from scipy.cluster.hierarchy import leaves_list

def get_clustergram_ordering(topic_dist, kmeans_cluster_labels):
    ## Get ordering for spectra
    ## This is obtained by performing hierarchical clustering within each KMeans cluster
    spectra_order = []
    for cl in sorted(set(kmeans_cluster_labels)):
        cl_filter = kmeans_cluster_labels==cl

        cl_dist = squareform(topics_dist[cl_filter, :][:, cl_filter])
        cl_dist[cl_dist < 0] = 0 #Rarely get floating point arithmetic issues
        cl_link = linkage(cl_dist, 'average')
        cl_leaves_order = leaves_list(cl_link)

        spectra_order += list(np.where(cl_filter)[0][cl_leaves_order])
    return(spectra_order)

def get_shuffle(num, period):
    ordered = np.arange(1, num+1)
    if period == 1:
        shuffled = ordered
    else:
        shuffled = [int(i/num)+(i%num) for i in range(1, num*period, period)]
    mapping = dict(zip(ordered, shuffled))
    return(mapping)

from cnmf import fast_euclidean
from sklearn.cluster import KMeans




## Repeat the clustering and consensus matrix plotting but this time outputting a labeled figure without the histogram

replicate_spectra_fn = cnmf_obj.paths['merged_spectra'] % 20
replicate_spectra = load_df_from_npz(replicate_spectra_fn)
l2_norms = (replicate_spectra**2).sum(axis=1).apply(np.sqrt)
l2_spectra = replicate_spectra.div(l2_norms, axis=0)
topics_dist = squareform(fast_euclidean(l2_spectra.values))
kmeans_model = KMeans(n_clusters=20, n_init=10, random_state=1)
kmeans_model.fit(l2_spectra)
kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)
spectra_order = get_clustergram_ordering(topics_dist, kmeans_cluster_labels)
topics_dist = topics_dist[spectra_order, :][:, spectra_order]
kmeans_cluster_labels = kmeans_cluster_labels.iloc[spectra_order]

## Permute the cluster order so that the colors aren't sequential
permmap = get_shuffle(20, 5)
kmeans_cluster_labels_perm = kmeans_cluster_labels.apply(lambda x: permmap[x])        
(heatmapax, topax, leftax, heatmapfig, heatmapcbar, fig) = plotting.labeled_heatmap(topics_dist, kmeans_cluster_labels_perm.values,
                                                                      1, figsize=(3.,3.), dpi=300)


heatmapax.set_xlabel('NMF Components', fontsize=12)
leftax.set_ylabel('NMF Components', fontsize=12)
topax.set_title('Pre-filtered Clustering', fontsize=13)
fig.savefig('S12b_VisualCortex_cNMF_Clustergram_Prefilter.pdf', dpi=200, pad_inches=.1, bbox_inches='tight')

from plotting import labeled_heatmap



#############################################################################


# Step 5 in paper code

import seaborn as sns
import glob
from scipy.stats import mannwhitneyu, fisher_exact, pearsonr
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
from cnmf import load_df_from_npz, save_df_to_npz

# Annotate the inferred cNMF components through comparison with the published clusters and the treatment status of the cells. Specifically, look for GEPs that are extensively used by cells in a particular cluster from Hrvatin Et. Al. or that correlate with treatment conditions
published_clusters = pd.read_csv('GSE102827/GSE102827_cell_type_assignments.csv', index_col=0)

usage_matrix = pd.read_csv('GSE102827_cNMF/GSE102827_cNMF.usages.k_20.dt_0_08.consensus.txt', sep='\t', index_col=0)
usage_matrix.columns = np.arange(1,21)
normalized_usage_matrix = usage_matrix.div(usage_matrix.sum(axis=1), axis=0)


## Get the % of cells of a cluster that have a >.25 usage of each GEP
usage_by_cluster = normalized_usage_matrix.groupby(published_clusters['celltype'], axis=0).apply(lambda x: (x>.25).mean())

## Order the GEPs to match up to the published clusters
cluster_order = ['ExcL23', 'ExcL4', 'ExcL5_1', 'ExcL5_2', 'ExcL5_3', 'ExcL6',
                 'Sub', 'Hip', 'RSP', 'Int_Cck', 'Int_Vip', 'Int_Npy', 'Int_Pv',
                 'Int_Sst_1', 'Int_Sst_2']
gep_order = np.arange(1,21)
gep_order = [16,17,5,3,11,14,12,18,6,8,7,4,10,2,9, 15, 1, 19, 20, 13]
usage_by_cluster = usage_by_cluster.loc[cluster_order, gep_order]

(fig,ax) = plt.subplots(1,1,figsize=(3,2), dpi=500)
sns.heatmap(usage_by_cluster, ax=ax)
ax.set_title('% Cells With GEP Usage > 25%\nby Published Cluster')
ax.set_ylabel('Clustered Cell-Type')
ax.set_xlabel('Inferred GEP')
ax.set_yticks(np.arange(15)+.5)
ax.set_yticklabels(cluster_order, fontsize=4)

ax.set_xticks(np.arange(20)+.5)
_ = ax.set_xticklabels(gep_order, fontsize=4)



########################################################
# Saving files

usage_matrix_2 = pd.read_csv('GSE102827_cNMF/GSE102827_cNMF.usages.k_20.dt_0_08.consensus.txt', sep='\t', index_col=0)
usage_matrix_2.columns = np.arange(1,21)
normalized_usage_matrix_2 = usage_matrix_2.div(usage_matrix_2.sum(axis=1), axis=0)

normalized_usage_matrix_2.to_csv('GSE102827_cNMF/GSE102827_cNMF.usages.k_20.dt_0_08.consensus_Normalized.csv')

## Get the % of cells of a cluster that have a >.25 usage of each GEP
usage_by_cluster_2 = normalized_usage_matrix_2.groupby(published_clusters['celltype'], axis=0).apply(lambda x: (x>.25).mean())

usage_by_cluster_2.to_csv('GSE102827_cNMF/GSE102827_cNMF_usage_by_cluster.csv')
usage_by_cluster_2 = usage_by_cluster_2.loc[cluster_order, gep_order]
usage_by_cluster_2.to_csv('GSE102827_cNMF/GSE102827_cNMF_usage_by_cluster_ordered.csv')

#########################################################


## Mapping from reordered GEPs to labels
geplabel_map = {1:'Exc. L2', 2:'Exc. L3', 3:'Exc. L4', 4:'Exc. L5-1', 5:'Exc. L5-2', 6:'Exc. L5-3', 7:'Exc. L6-1', 
8:'Exc. L6-2', 9:'Sub', 10:'Hip', 11:'Int. Cck/Vip', 12:'Int. Npy', 13:'Int. Pv', 14:'Int. Sst-1', 15:'Int. Sst-2',
           16:'ERP', 17:'LRP-S', 18:'LRP-D', 19:'Syn', 20:'Other'}

## Mapping from reordered GEPs to labels
geplabels = ['Exc. L2', 'Exc. L3', 'Exc. L4', 'Exc. L5-1', 'Exc. L5-2', 'Exc. L5-3', 'Exc. L6-1', 
'Exc. L6-2', 'Sub', 'Hip', 'Int. Cck/Vip', 'Int. Npy', 'Int. Pv', 'Int. Sst',
           'ERP', 'LRP-S', 'LRP-D', 'Syn', 'NS', 'Other']



## Reorder the usages and then relabel them
normalized_usage_matrix_labeled = normalized_usage_matrix_2.loc[:, gep_order]
normalized_usage_matrix_labeled.columns = geplabels

normalized_usage_matrix_labeled.to_csv('GSE102827_cNMF/GSE102827_cNMF.usages.k_20.dt_0_08.consensus_Normalized_Labeled.csv')


## Relabel the gene scores that were also output by cNMF

genescores = pd.read_csv('GSE102827_cNMF/GSE102827_cNMF.gene_spectra_score.k_20.dt_0_08.txt',
                sep='\t', index_col=0).T
genescores.columns = np.arange(1,21)
genescores_labeled = genescores.loc[:, gep_order]
genescores_labeled.columns = geplabels
genescores_labeled.to_csv('GSE102827_cNMF/GSE102827_cNMF.gene_spectra_score.k_20.dt_0_08.labeled.df.csv')
genescores_labeled.head()




## Create a column of updated labels for plotting
clabels = ['ExcL23', 'ExcL4', 'ExcL5_1', 'ExcL5_2', 'ExcL5_3', 'ExcL6', 'Sub', 'Hip', 'RSP', 'Int_Cck', 'Int_Vip', 'Int_Npy', 'Int_Pv', 'Int_Sst_1', 'Int_Sst_2']
clabelsfix = ['Exc. L2,3', 'Exc. L4', 'Exc. L5-1', 'Exc. L5-2', 'Exc. L5-3', 'Exc. L6', 'Sub', 'Hip', 'RSP', 'Int. Cck', 'Int. Vip', 'Int. Npy', 'Int. Pv', 'Int. Sst-1', 'Int. Sst-2']
published_clusters = published_clusters.loc[normalized_usage_matrix_labeled.index, :]
labmap = dict(zip(clabels, clabelsfix))
published_clusters['celltype_forplot'] = published_clusters['celltype'].apply(lambda x: labmap[x])

usage_info = normalized_usage_matrix_labeled.unstack().reset_index()
usage_info.columns = ['GEP', 'Cell', 'Usage']
usage_info = pd.merge(left=usage_info, right=published_clusters[['celltype_forplot', 'stim']],
                     left_on='Cell', right_index=True, how='left')





#################################################################

identity_programs =['Hip',
 'Sub',
 'Exc. L2',
 'Exc. L3',
 'Exc. L4',
 'Exc. L5-1',
 'Exc. L5-2',
 'Exc. L5-3',
 'Exc. L6-1',
 'Exc. L6-2',
 'Int. Sst',
 'Int. Pv',
 'Int. Cck/Vip',
 'Int. Npy']

activity_programs = ['ERP', 'LRP-S', 'LRP-D', 'Syn', 'NS', 'Other']

n_identity_programs = len(identity_programs)
n_activity_programs = len(activity_programs)
n_geps = normalized_usage_matrix_labeled.shape[1]

#################################################################

# heatmap

## Get an approximate assignment of each cell to a dominant GEP to be used for plotting

identity_gep_usage = normalized_usage_matrix_labeled.loc[:, ~normalized_usage_matrix_labeled.columns.isin(activity_programs)].copy()
identity_gep_usage = identity_gep_usage.div(identity_gep_usage.sum(axis=1), axis=0)

cell_max_identity_usage = identity_gep_usage.max(1)
cell_max_identity_gep = identity_gep_usage.idxmax(1)

overall_cell_max_usage = normalized_usage_matrix_labeled.max(1)
overall_cell_max_gep = normalized_usage_matrix_labeled.idxmax(1)

# Get which cells are assigned to which identity maximum GEP
per_topic_filter = {}
for ti in sorted(set(cell_max_identity_gep)):
    per_topic_filter[ti] = (cell_max_identity_gep==ti)
    
#Find the second highest topic for each cell (Excluding the first, but including activity programs)
notop_raw_gep_usage = normalized_usage_matrix_labeled.values.copy()
for ti in range(n_geps):
    if ti in per_topic_filter:
        notop_raw_gep_usage[per_topic_filter[ti], ti-1] = 0
second_highest_gep = notop_raw_gep_usage.argsort(1)[:, -1]








fig = plt.figure(figsize=(3, 4), dpi=600)
gs = gridspec.GridSpec(2, n_identity_programs, fig,
#                        0.18, 0.07, 0.98, 0.93,
                       0.22, 0.11, 0.95, 0.92,
                       hspace=0.03, wspace=0.1,
                       height_ratios = [n_identity_programs, n_activity_programs]
                      )

ax = fig.add_subplot(gs[:n_identity_programs], 
                     title='Cells ',
                     frameon=False, 
                     xticks=[], yticks=[])


for pi,ti in enumerate(identity_programs):

    ax = fig.add_subplot(gs[0, pi], title='',
          xscale='linear', yscale='linear', ylabel='',
          frameon=False, xticks=[], yticks=[])
    
    i_order = np.where(per_topic_filter[ti])[0][second_highest_gep[per_topic_filter[ti]].argsort()]
    im = ax.imshow(normalized_usage_matrix_labeled.iloc[i_order, ].loc[:, identity_programs].T,
              aspect='auto',
                   interpolation='nearest', cmap="Reds", vmin=0, vmax=1, 
              rasterized=True)
    
    if ax.is_first_col():
        ax.set_yticks(range(len(identity_programs)))
        ax.set_yticklabels(identity_programs)  
        
        
    ax = fig.add_subplot(gs[1, pi], title='',
          xscale='linear', yscale='linear', ylabel='',
          frameon=False, xticks=[], yticks=[])
    
    i_order = np.where(per_topic_filter[ti])[0][second_highest_gep[per_topic_filter[ti]].argsort()]
    im = ax.imshow(normalized_usage_matrix_labeled.iloc[i_order, ].loc[:, activity_programs].T,
              aspect='auto',
                   interpolation='nearest', cmap="Reds", vmin=0, vmax=1, 
              rasterized=True)
    
    if ax.is_first_col():
        ax.set_yticks(range(len(activity_programs)))
        ax.set_yticklabels(activity_programs) 
        

cax2 = fig.add_axes([0.24, 0.04, 0.12, 0.015],
                xscale='linear', yscale='linear',
                xlabel='', ylabel='', frameon=True, )
    
with mpl.rc_context({"axes.linewidth": 0.5}):
    cb = fig.colorbar(im, cax=cax2, orientation='horizontal', ticks=[0, 0.5, 1.0])
    # cb.ax.invert_xaxis()
#     cb.set_label('GEP usage', fontsize=5)
    cb.ax.xaxis.set_label_position('top')
    cb.set_ticklabels(['0', '.5', '1'])
    cb.ax.tick_params(labelsize=5, width=0.5, length=3)
    cax2.text(1.1, 0.5, 'GEP usage by cells', va='center', fontsize=6)
            

fig.savefig('GSE102827_cNMF/GSE102827_Gep_Usage.pdf', dpi=600, bbox_inches='tight')










maxt = normalized_usage_matrix_labeled.idxmax(axis=1)
maxv = normalized_usage_matrix_labeled.max(axis=1)
comb = pd.concat([maxt, maxv],axis=1)
comb.columns = ['max GEP', 'max val']
(fig,ax) = plt.subplots(1,1, figsize=(10,3), dpi=600)
sns.boxplot(x='max GEP', y='max val', data=comb, ax=ax, order=identity_programs+activity_programs)
ax.set_ylabel('Usage', fontsize=10)
ax.set_xlabel('Max GEP', fontsize=10)



############################################################################

#Usage of activity programs by stimulation status

activity_programs = ['ERP', 'LRP-S', 'LRP-D', 'Syn', 'NS', 'Other' ]
exctypes = [ 'Exc. L2', 'Exc. L3', 'Exc. L4', 'Exc. L5-1', 'Exc. L5-2', 'Exc. L5-3',
             'Exc. L6-1', 'Exc. L6-2',]
othertypes = ['Sub','Hip']
interneuron = [ 'Int. Sst', 'Int. Pv', 'Int. Cck/Vip', 'Int. Npy']

activity_gep_usage_by_class = pd.merge(left=normalized_usage_matrix_labeled[activity_programs],
        right=published_clusters[['stim']], left_index=True, right_index=True)

ind = activity_gep_usage_by_class.index
identity_cols = [x for x in normalized_usage_matrix_labeled if x not in activity_programs]
activity_gep_usage_by_class['max_identity_gep'] = normalized_usage_matrix_labeled.loc[ind,identity_cols].idxmax(1).values
activity_gep_usage_by_class['max_identity_gep_usage'] = normalized_usage_matrix_labeled.loc[ind,identity_cols].max(1).values



activity_gep_usage_by_class['max_identity_class'] = activity_gep_usage_by_class['max_identity_gep']
identiy = activity_gep_usage_by_class['max_identity_class']
activity_gep_usage_by_class.loc[identiy.isin(othertypes), 'max_identity_class'] = 'Non-VC Contaminant'
activity_gep_usage_by_class.loc[identiy.isin(interneuron), 'max_identity_class'] = 'Interneuron'

activity_gep_usage_by_class.to_csv('GSE102827_cNMF/GSE102827_cNMF.activity_GEP_usage_by_class.csv')


used_groups = exctypes + ['Non-VC Contaminant', 'Interneuron']
used_stim = ['0h', '1h', '4h']


###########################################################################

# Activity program usage boxplot

import matplotlib as mpl
import palettable
import matplotlib.patches as patches

label_size = 8

core_colors = type('CoreColors', (), {})

cnames = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'brown', 'pink', 'grey']

def to_array_col(color):
    return np.array(color)/255.

for cname,c in zip(cnames, palettable.colorbrewer.qualitative.Set1_9.colors):
    setattr(core_colors, cname, np.array(c)/255.)
    
for cname, c in zip(['blue', 'green', 'red', 'orange', 'purple'],
                    palettable.colorbrewer.qualitative.Paired_10.colors[::2]):
    setattr(core_colors, 'pale_'+cname, np.array(c)/255.)
    
core_colors.teal = to_array_col(palettable.colorbrewer.qualitative.Set2_3.colors[0])
core_colors.brown_red = to_array_col(palettable.colorbrewer.qualitative.Dark2_3.colors[1])

# core_colors.light_grey = to_array_col(palettable.colorbrewer.qualitative.Set2_8.colors[-1])
core_colors.light_grey = to_array_col(palettable.tableau.TableauLight_10.colors[7])

core_colors.royal_green = to_array_col(palettable.wesanderson.Royal1_4.colors[0])






fig = plt.figure(figsize=(3, 4), dpi=600)
gs = gridspec.GridSpec(6, 1,fig,
                       0.20, 0.11, 0.98, 0.92,
                       hspace=.12)
    
col = 'Big GEP Group'
clabels = exctypes + ['Non-VC Contaminant', 'Interneuron']
nclusters = len(clabels)

activity_names = ['Early\nresponse\n(ER)', 'Superficial late\nresponse\n(LR-S)',
               'Deep late\nresponse\n(LR-D)', 'Synapto-\ngenisis\n(Syn)',
                  'Neuro-\nsecretory\n(NS)',
               'Other\nprogram\n(Other)']

activity_id2name = dict(zip(activity_programs, activity_names))



position = (np.arange(len(clabels))%3 - 1)*0.3
n_groups = len(used_groups)*len(used_stim)
positions = np.arange(n_groups) + (np.arange(n_groups)%3 - 1)*(-0.3)

stim_facecolor = {'4h': core_colors.pale_blue, '1h': core_colors.pale_green, '0h': core_colors.pale_orange}
stim_edgecolor = {'4h': core_colors.blue, '1h': core_colors.green, '0h': core_colors.orange}
subdat = activity_gep_usage_by_class
for (i,act_gep) in enumerate(activity_programs):
    ax = fig.add_subplot(gs[i,0], frameon=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    grp_def = []
    grouped_usages = []
    for grp in used_groups:
        for stim in used_stim:
            grp_def += [(grp, stim)]
            grouped_usages.append(subdat[(subdat['stim']==stim) & (subdat['max_identity_class']==grp)][act_gep].values)

    bp = ax.boxplot(grouped_usages, sym='o', notch=False, zorder=10, whis=[5,95], positions=positions,
                    patch_artist=True)
    for el in ['whiskers', 'caps']:
        for li,line in enumerate(bp[el]):
            line.set_color(stim_edgecolor[grp_def[int(np.floor(li/2))][1]])
            line.set_linewidth(0.5)
            line.set_zorder(-10)
            line.set_clip_on(False)
            
    for li,line in enumerate(bp['boxes']):
        line.set_linewidth(0.5)
        line.set_zorder(-10)
        line.set_facecolor(stim_facecolor[grp_def[li][1]])
        line.set_edgecolor(stim_edgecolor[grp_def[li][1]])
        line.set_clip_on(False)
    for li,line in enumerate(bp['medians']):
        line.set_zorder(-1)
        line.set_color(stim_edgecolor[grp_def[li][1]])
    for li,line in enumerate(bp['fliers']):
        line.set_markeredgecolor('none')
        line.set_markerfacecolor(stim_facecolor[grp_def[li][1]])
        line.set_alpha(0.3)
        line.set_markersize(1)
        line.set_rasterized(True)
    ax.set_xticks([])
    ax.set_yticks([0, .25, .5, .75])
    ax.set_yticklabels([0, 25, 50, 75])
    ax.set_ylim([0, .75])
    ax.set_xlim(-0.5, len(grp_def)-0.5)
    for pi in range(1, len(used_groups), 2):
        ax.add_patch(
            patches.Rectangle(
                (pi*3-0.5, 0.), 3.0, 0.75, 
                facecolor="#f0f0f0", edgecolor='none', zorder=-20,
                )
            )
        
    ax.set_ylabel(activity_id2name[act_gep], fontsize=7)
    
    if ax.is_last_row():
        ax.set_xticks(range(1, len(grp_def), 3))
        ax.set_xticklabels(['Exc.\nL2', 'Exc.\nL3', 'Exc.\nL4', 'Exc.\nL5-1', 'Exc.\nL5-2',
                            'Exc.\nL5-3', 'Exc.\nL6-1', 'Exc.\nL6-2', 'Non-\nvisual\ncortex', 'Inter-\nneuron']
                           ,fontsize=5)
        ax.set_xlabel('Cells from identity program', fontsize=7)
    if ax.is_first_row():
        ax.set_title('Activity program usage (%)')
        
        stim_legend = [patches.Patch(facecolor=stim_facecolor['0h'], linewidth=0.5,
                                     edgecolor=stim_edgecolor['0h'], label='0h'),
                patches.Patch(facecolor=stim_facecolor['1h'], linewidth=0.5,
                              edgecolor= stim_edgecolor['1h'], label='1h post-stim'),
                patches.Patch(facecolor=stim_facecolor['4h'], linewidth=0.5,
                              edgecolor= stim_edgecolor['4h'], label='4h post-stim'),
                       ]
        legend = ax.legend(handles=stim_legend, fontsize=5, loc=(.75,.45))
        legend.get_frame().set_linewidth(0.5)


fig.savefig('GSE102827_cNMF/GSE102827_Activity_Gep_Boxplots.pdf', dpi=600, bbox_inches='tight')



















