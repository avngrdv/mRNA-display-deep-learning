# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:34:18 2021
@author: Alex Vinogradov
"""

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from config import constants
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

class SequencingData:
    '''
    Just a container for FastqParser-related plotting funcs
    '''
    
    def L_distribution(X, Y, where, basename):
    
        fig = plt.figure(figsize=(18, 6), dpi=300)
        ax = fig.add_subplot(111)
        plt.bar(X, Y, color='#0091b5')
    
        ax.set_ylim(0, 1.02*np.max(Y))
        ax.set_xlim(np.min(X), np.max(X)+1)
        ax.set_xticks(np.linspace(np.min(X), np.max(X)+1, 10))
        ax.set_xticklabels(np.linspace(np.min(X), np.max(X)+1, 10, dtype=int))
        
        ax.set_xlabel('Sequence length', fontsize=30)
        ax.tick_params(axis='both', which='major',  labelsize=25)                                                 
        ax.set_ylabel('Count', fontsize=30)                     
    
        title = f'Distribution of sequence lengths in {where} dataset'
        ax.set_title(title, fontsize=34, y=1.04)
                                              
        #save png and svg, and close the file
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()    

    def dataset_convergence(C, shannon, where, basename):
    
        y = np.sort(C)
        x = 100 * np.divide(np.arange(y.size), y.size)
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)    
        plt.plot(x, y, lw=2.5, c='#3b61b1', antialiased=True)

        ax.set_xlim(0, 101)
        ax.set_xticks(np.arange(0, 125, 25))
        
        ax.set_ylabel(f'{where} sequence count', fontsize=14)
        ax.set_xlabel('Sequence percentile', fontsize=14)
        ax.set_title(f'Sequence-level convergence of {where} dataset', fontsize=16)
        plt.text(x=2, 
                 y=y.max(),
                 s=f'normalized Shannon entropy: {shannon:1.4f}', 
                 size=12,
                 horizontalalignment='left',
                 verticalalignment='center',)
            
        plt.grid(lw=0.5, ls='--', c='slategrey', 
                 dash_capstyle='round', dash_joinstyle='round',
                 antialiased=True, alpha=0.2)  
        
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()      
            
    def conservation(conservation, where, basename):
        
        fig = plt.figure(figsize=(18, 6), dpi=300)
        ax = fig.add_subplot(111)
        plt.plot(conservation, lw=3.5, c='#3b61b1')
    
        y_lim = np.ceil(np.max(conservation))
        ax.set_ylim(0, y_lim)
        ax.set_xlim(0, len(conservation))
        ax.set_xticks(np.linspace(0, len(conservation), 10))
        ax.set_xticklabels(np.linspace(0, len(conservation), 10, dtype=int))
        
        ax.set_xlabel('Sequence index', fontsize=30)
        ax.tick_params(axis='both', which='major',  labelsize=25)                                                 
        ax.set_ylabel('Conservation, bits', fontsize=30)                     
    
        title = f'Token-wise sequence conservation plot for {where} dataset'
        ax.set_title(title, fontsize=34, y=1.04)
                                              
        #save png and svg, and close the file
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()


    def tokenwise_frequency(freq, yticknames, where, loc, basename):
    
        if where == 'dna':
            ylabel = 'Base'

        if where == 'pep':
            ylabel = 'Amino acid'
            
        figsize = (1 + freq.shape[1] / 2, freq.shape[0] / 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(freq))
        c = ax.pcolormesh(freq, cmap=plt.cm.Blues, norm=norm, edgecolors='w', linewidths=4)
        cbar = fig.colorbar(c, ax=ax)
    
        cbar.ax.set_ylabel("frequency", rotation=-90, va="bottom", fontsize=22)
        cbar.ax.tick_params(labelsize=20)    
    
        #set ticks
        ax.set_xticks(np.arange(freq.shape[1])+0.5)
        ax.set_yticks(np.arange(freq.shape[0])+0.5)
        ax.set_xticklabels(np.arange(freq.shape[1])+1)
        ax.set_yticklabels(yticknames)
    
        #set labels
        ax.set_xlabel(f'Position inside library region(s) {loc}', fontsize=21)
        ax.set_ylabel(ylabel, fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f'Position-wise frequency map for {where} dataset', fontsize=25)
        
        #save png and svg, and close the file
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()    

    def Q_score_summary(avg, std, loc, basename):
         
        fig = plt.figure(figsize=(18, 6), dpi=300)
        ax = fig.add_subplot(111)
        plt.plot(avg, lw=4, c='#3b61b1')
        plt.plot(avg+std, lw=1, c='#0091b5')
        plt.plot(avg-std, lw=1, c='#0091b5')
        ax.fill_between(np.arange(len(avg)), avg-std, avg+std, color='#0091b5', alpha=0.15)
    
        ax.set_ylim(0, 53)
        ax.set_xlim(0,len(avg))
        ax.set_xticks(np.linspace(0, len(avg), 10))
        ax.set_xticklabels(np.linspace(0, len(avg), 10, dtype=int))
  
        ax.set_xlabel(f'{loc} region(s) index', fontsize=30)
        ax.tick_params(axis='both', which='major',  labelsize=25)                                               
        ax.set_ylabel('Q, average log score', fontsize=30)                     
    
        title = 'Q-score plot'
        ax.set_title(title, fontsize=34, y=1.04)
                                              
        #save png and svg, and close the file
        svg = basename + '.svg'
        png = basename + '.png'
        fig.savefig(svg, bbox_inches = 'tight')
        fig.savefig(png, bbox_inches = 'tight')
        plt.close()   


#plotter for "../interrogators/ig_to_structure.py"
def attribution_colorbar(norm, cmap, figname):
    
    fig, ax = plt.subplots(1, 1, figsize=(0.3, 5), dpi=300)
    
    mpl.colorbar.ColorbarBase(ax,
                              orientation='vertical', 
                              cmap=cmap,
                              norm=norm,
                              label='Attribution magnitude')
    
    fig.savefig(figname, bbox_inches='tight')
    plt.close()
    return

#plotter for "../interrogators/virtual_mutagenesis.py"
def virtual_mutagenesis(proba, parent, basename):
    
    shape = 4 + (np.array(proba.T.shape) / 2)
    
    fig, ax = plt.subplots(1, 1, figsize=shape, dpi=300)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    cmap = sns.color_palette("mako", as_cmap=True)
    
    c = ax.pcolor(proba, cmap=cmap, norm=norm,  
                  edgecolors='w', linewidths=4)
    
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("CNN calls", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(proba.shape[1])+0.5)
    ax.set_yticks(np.arange(proba.shape[0])+0.5)   
    ax.set_xticklabels(parent)
    ax.set_yticklabels(constants.aas)

    #set labels
    ax.set_xlabel('Wild type amino acid', fontsize=21)
    ax.set_ylabel('Mutated to', fontsize=21)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_title(f'Virtual mutagenesis for {"".join(parent)}', fontsize=24)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return

def positional_epistasis(epi, pos1, pos2, basename):
    '''
    A 2D map of epistatic interactions between amino acids in pos1 and pos2
    Used to make Fig. S7

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
                           
             pos1, pos2:   int
                           
    '''   
     
    epi = epi[pos1, pos2]
    fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=300)
    
    scale = np.max([np.abs(np.nanmin(epi)), np.abs(np.nanmax(epi))])
    
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    c = ax.pcolor(epi, cmap=plt.cm.RdBu, norm=norm, edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Epistasis log score", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(epi.shape[1])+0.5)
    ax.set_yticks(np.arange(epi.shape[0])+0.5)
    ax.set_xticklabels(constants.aas)
    ax.set_yticklabels(constants.aas)

    #set labels
    x_label = 'Amino acid in position ' + str(pos2+1)
    ax.set_xlabel(x_label, fontsize=25)
    
    y_label = 'Amino acid in position ' + str(pos1+1)
    ax.set_ylabel(y_label, fontsize=25)
    
    ax.tick_params(axis='both', which='major', labelsize=21)
    title = 'Epistasis between positions ' + str(pos1+1) + ' and ' + str(pos2+1)
    ax.set_title(title, fontsize=27)
    
    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    plt.close()  
    

def epistasis_bw_positions(epi, basename):
    '''
    Reduce epi array to (seq_len, seq_len) by averaging along the last 
    two axes and plot the the result.
    Used to make Fig. 3c and 5f

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library                   
    '''   
    
    epi = np.nanmean(np.abs(epi), axis=(2,3))

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6), dpi=300) 
    norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
    c = ax.pcolor(epi, cmap=mpl.cm.cividis, norm=norm, edgecolors='w', linewidths=4)    
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("abs(epi) score", rotation=-90, va="bottom", fontsize=21)
    cbar.ax.tick_params(labelsize=21)    

    for y in range(epi.shape[0]):
        for x in range(epi.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % epi[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                 )

    #set ticks
    ax.set_xticks(np.arange(epi.shape[1])+0.5)
    ax.set_yticks(np.arange(epi.shape[0])+0.5)
    ax.set_xticklabels(np.arange(epi.shape[0])+1)
    ax.set_yticklabels(np.arange(epi.shape[0])+1)

    #set labels
    ax.set_xlabel('Variable region position', fontsize=21)
    ax.set_ylabel('Variable region position', fontsize=21)
    
    ax.tick_params(axis='both', which='major', labelsize=21)
    title = 'Average positional epistasis'
    ax.set_title(title, fontsize=23)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    return
    
def pep_epistatic_interactions(epi, pep, basename):
    '''
    Plot all pairwise  epistatic interactions for peptide pep.
    Used to make Fig. 3e and S15b

        Parameters:
                    epi:   4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
                           
                    pep:   peptide to the computation (1D np array, dtype=int)
    '''   
    
    fig, ax = plt.subplots(1, 1, figsize=(13, 7), dpi=300)
    
    ax.set_xlim(-0.5, pep.size - 0.5)
    ax.set_ylim(-0.5, np.divide(pep.size - 1, 2) + 0.5)
    
    rel_t = np.zeros((pep.size, pep.size))
    for pos1 in range(pep.size):
        for pos2 in range(pos1 + 1, pep.size):
            rel_t[pos1, pos2] = epi[pos1, pos2, pep[pos1], pep[pos2]]  
    
    #cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True)
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    scale = np.max([np.abs(np.nanmin(rel_t)), np.abs(np.nanmax(rel_t))])
    norm = mpl.colors.Normalize(vmin=-scale, vmax=scale)
    
    def arc(x1, x2, n=1000):
        #n - number of points to approximate the arc with
        x0 = np.divide(x1 + x2, 2)
        r = x2 - x0
        x = np.linspace(x1, x2, num=n)
        y = np.sqrt(r**2 - (x - x0)**2)
        return x, y

    for pos1 in range(pep.size):
        for pos2 in range(pos1 + 1, pep.size):
            x, y = arc(pos1, pos2)
            epi = rel_t[pos1, pos2]
            lw = 20 * np.power(np.divide(np.abs(epi), scale), 0.67)
            alpha = 1 * np.power(np.divide(np.abs(epi), scale), 0.5)
            
            ax.plot(x, y, color=cmap(norm(epi)), linewidth=lw, alpha=alpha)
    
    ax.scatter(np.arange(pep.size), np.zeros(pep.size,), s=700, alpha=1, color=cmap(norm(0)), zorder=10)
    ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, pad=0.01)
    cbar.ax.set_ylabel("Epistasis log score", rotation=-90, va="bottom", fontsize=18)
    cbar.ax.tick_params(labelsize=16)        
      
    seq = [constants.aas[x] for x in pep]
    for x in np.arange(pep.size):
        
        ax.text(x, 
                -0.1,
                seq[x],
                zorder=20, 
                fontsize=20,
                fontweight='bold', 
                color=(0.33, 0.33, 0.33, 1),
                ha='center')
        
    seq = ''.join(seq)
    ax.set_title(f'Position-wise epistasis in peptide {seq}', fontsize=18)

    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    return

def IG_attributions(igs, sequence, mod, basename):
    '''
    Used to make Fig. 4a, 4e, S8-10
    '''
    
    import tensorflow as tf
    fig = plt.figure(figsize=(18, 3), dpi=300)
    
    ax1 = plt.subplot2grid((1, 64), (0, 2), rowspan=1, colspan=62)
    ax2 = plt.subplot2grid((1, 64), (0, 0), rowspan=1, colspan=1)
    
    cmap = plt.cm.cividis
    c = ax1.pcolor(igs[::-1], cmap=cmap, edgecolors='w', linewidths=0.3)
    

    one_dim_igs = tf.reduce_sum(igs, axis=-1).numpy()
    ax2.pcolor(one_dim_igs[None, ::-1].T, cmap=cmap, edgecolors='w', linewidths=0.3)

    cbar = fig.colorbar(c, ax=ax1, pad=0.01)
    cbar.ax.set_ylabel("attributions", rotation=-90, va="bottom", fontsize=16) 

    #set ticks
    ax1.set_xticks(np.arange(0.5, igs.shape[1] + 0.5, 20))
    ax1.set_xticklabels(np.arange(1, igs.shape[1]+1, 20))
    ax1.set_xlabel('Fingerprint index', fontsize=20)
    ax1.yaxis.set_visible(False)
    
    
    ax2.set_yticks(np.arange(igs.shape[0])+0.5)
    ax2.set_yticklabels(sequence[::-1])
    ax2.xaxis.set_visible(False)
    
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax1.set_title(f"IG attributions, peptide: {''.join(sequence)}, yield: {mod[0]:.3f}, cnn call: {mod[1]:.3f}", fontsize=22)
    
    #save png and svg, and close the file
    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    plt.close()    
    
   
def cum_proba_distribution(proba_bundle, basename):
    '''
        Parameters:
           proba_bundle:   list of tuples (probabilities for a given sample, sample name)
                           Overlay plots for each tuple
                           
    '''   
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    c = np.linspace(0, 1, len(proba_bundle) + 2)
    cmap = mpl.cm.get_cmap('cividis')
    
    for i, sample in enumerate(proba_bundle):
        y = sample[0]
        y.sort()
        x = np.linspace(0, 100, y.size)
        label = f'{sample[1]}; avg call: {np.mean(y):.2f}'
        plt.plot(x, y, lw=4, c=cmap(c[i+1]), label=label, alpha=0.8)

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.round(np.linspace(0, 100, 6), decimals=0))
    ax.set_xticklabels(np.linspace(0, 100, 6))
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels(np.round(np.linspace(0, 1, 6), decimals=1))
    
    ax.set_xlabel('Peptides, percentile', fontsize=30)
    ax.tick_params(axis='both', which='major',  labelsize=25)                                               
    ax.set_ylabel('Model call', fontsize=30)                     
    plt.legend(frameon=False, loc='upper left', prop={'size': 10})


    title = 'Cumulative model probability calling'
    ax.set_title(title, fontsize=34, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()  


def s_distribution_hist(s_scores, basename):
    '''
    Fig. 2e, S13b
        
        Parameters:
               s_scores:   list of tuples where each tuple corresponds to a sample:
                           (np ndarray of s scores, sample name)
                           
    '''  
  
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    s_min = 999  
    s_max = -999
     
    ct = ['#0091b5' ,'#db5000', '#3b61b1']
    
    for i,sample in enumerate(s_scores):
        data = sample[0]
        
        if data.max() > s_max:
            s_max = data.max()
            
        if data.min() < s_min:
            s_min = data.min()        
    
        plt.hist(data, bins=100, color=ct[i], label=sample[1], alpha=0.5, density=True)

    s_min = -17.4

    ax.set_xlim(1.05*s_min, 1.05*s_max)
    ax.set_xticks(np.linspace(s_min, s_max, 6))
    ax.set_xticklabels(np.round(np.linspace(s_min, s_max, 6), decimals=1))
        
    ax.set_xlabel('S score', fontsize=24)                                          
    ax.set_ylabel('Distribution density', fontsize=24)                     
    ax.tick_params(axis='both', which='major',  labelsize=20)     
    plt.legend(frameon=False, loc='upper left', prop={'size': 10})

    title = 'S score distribution density'
    ax.set_title(title, fontsize=24, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()    

def cum_e_distribution(s_scores, basename):
    '''
    Fig. S2
        
        Parameters:
               s_scores:   list of tuples where each tuple corresponds to a sample:
                           (np ndarray of s scores, sample name)
                           
    '''    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    ymin = 999
    ymax = -999
    c = np.linspace(0, 1, len(s_scores) + 2)
    cmap = mpl.cm.get_cmap('cividis')
    
    for i,sample in enumerate(s_scores):
        y = sample[0]
        x = np.linspace(0, 100, y.size)
        
        if y.max() > ymax:
            ymax = y.max()
        
        if y.min() < ymin:
            ymin = y.min()
            
        plt.plot(x, y, lw=4, c=cmap(c[i+1]), label=sample[1], alpha=0.8)

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 6))
    ax.set_xticklabels(np.linspace(0, 100, 6))
    
    ax.set_ylim(1.05*ymin, 1.05*ymax)
    ax.set_yticks(np.linspace(ymin, ymax, 6))
    ax.set_yticklabels(np.round(np.linspace(ymin, ymax, 6), decimals=1))
    
    ax.set_xlabel('Peptides, percentile', fontsize=30)
    ax.tick_params(axis='both', which='major',  labelsize=25)                                               
    ax.set_ylabel('S score', fontsize=30)                     
    plt.legend(frameon=False, loc='lower right', prop={'size': 10})


    title = 'S cumulative statistic'
    ax.set_title(title, fontsize=34, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()            

def acc_vs_round(rounds, acc, basename):
    '''
    Fig. 2f
        
        Parameters:
               rounds:   1D np.ndarray
                  acc:   1D np.ndarray holding model accuracy values
                         acc.size = rounds.size
    '''  
    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)

    plt.scatter(rounds, acc, s=20, alpha=0.7, color='slateblue', edgecolors='none')    

    ax.set_xlim(0.5, 6.5)
    ax.set_xticks(np.arange(1, 7, 1))

    ax.set_ylim(0.5, 2.5)
    
    ytick = np.array([0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995])    
    ax.set_yticks(-np.log10(1 - ytick))
    ax.set_yticklabels(ytick)

    ax.set_xlabel('Selection round', fontsize=20)
    ax.tick_params(axis='both', which='major',  labelsize=16)                                               
    ax.set_ylabel('-Lg(1-acc)', fontsize=20)                     

    title = 'Model accuracy vs. selection round'
    ax.set_title(title, fontsize=24, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()   


def acc_vs_data(size, acc, basename):
    '''
    Fig. 2g, 5d
        
        Parameters:
                 size:   1D np.ndarray
                  acc:   1D np.ndarray holding model accuracy values
                         acc.size = size.size
    '''  
    
    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    plt.scatter(size, acc, s=100, alpha=0.85, marker='d', edgecolors='none', color='steelblue')

    ax.set_ylim(0.3, 2.6)
    ax.set_xlim(1.5, 7.5)
    ytick = np.array([0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997])    
    ax.set_yticks(-np.log10(1 - ytick))
    ax.set_yticklabels(ytick)
    
    ax.set_xlabel('Lg(number of training samples', fontsize=20)
    ax.tick_params(axis='both', which='major',  labelsize=16)                                               
    ax.set_ylabel('-Lg(1-acc)', fontsize=20)                     

    title = 'Model accuracy vs. #training samples'
    ax.set_title(title, fontsize=24, y=1.04)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()   

def pep_yield_vs_prediction(y, proba, basename):
    '''
    Fig. 2i, 5e
        
        Parameters:
                    y:   1D np.ndarray, real peptide modification efficiencies
                proba:   1D np.ndarray, model calls for the same peptides
    ''' 

    fig = plt.figure(figsize=(16, 3), dpi=300)
    ax = fig.add_subplot(111)

    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    norm = mpl.colors.Normalize(vmin = 1.05 * np.min(proba - y), 
                                vmax = 1.05 * np.max(proba - y))

    for i in range(y.size):
        
        y1 = y[i]
        y2 = proba[i]
        if y2 != y1:
            plt.plot((i, i), (y1, y2), color=cmap(norm(y2-y1)), lw=4, alpha=0.7)

    plt.scatter(np.arange(y.size), y, s=100, marker='o', edgecolors='none', color='#3b61b1', zorder=100)
    plt.scatter(np.arange(proba.size), proba, s=100, marker='o', edgecolors='none', color='#0091b5', zorder=100)
        
    ax.set_xlim(-1, y.size+1)
    ax.set_ylim(-0.05, 1.05)
    
    ax.set_xticks(np.linspace(1, y.size, num=17))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(np.linspace(1, y.size, num=17))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, pad=0.01)

    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()      


def pep_yield_vs_s(y, S, basename):
    '''
    Fig. 3a, 5g
        
        Parameters:
                    y:   1D np.ndarray, real peptide modification efficiencies
                    S:   1D np.ndarray, S scores for the same peptides
    ''' 
    
    ind = np.argsort(S)
    S = S[ind]
    y = y[ind]

    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    plt.scatter(S, y, s=100, marker='o', edgecolors='none', color='#3b61b1', zorder=100)

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6))

    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()      


def e_vs_proba_v2(S, p, basename):
    '''
    Fig. 3b, 5h
        
        Parameters:
                    S:   1D np.ndarray, S scores for a peptide dataset
                    p:   1D np.ndarray, model probability calls for the same peptides
    ''' 
    
    from scipy.stats import binned_statistic
    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax1 = fig.add_subplot(111)

    std, bin_edges, _ = binned_statistic(S, p, statistic='std', bins=100)
    mean, bin_edges, _ = binned_statistic(S, p, statistic='mean', bins=100)
    x = bin_edges[:-1]  + np.diff(bin_edges) / 2
    
    ax1.hist(S, bins=bin_edges, color='#0091b5', alpha=0.3, density=True)
    ax2 = ax1.twinx()

    ax2.scatter(x, mean, color='#3b61b1', s=80, edgecolors='none', alpha=0.8)
    ax2.scatter(x, mean+std,color='#db5000', s=30, edgecolors='none', alpha=0.6, marker='v')
    ax2.scatter(x, mean-std, color='#db5000', s=30, edgecolors='none', alpha=0.6, marker='^')
    ax2.fill_between(x, mean-std, mean+std, color='#3b61b1', alpha=0.1)
    
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks(np.linspace(0, 1, 6))                                        
    ax2.set_ylabel('Model calls', fontsize=24)                     

    ax1.set_xlim(-12.8, 7.8)
    ax1.set_xticks(np.linspace(-17.4, 8.2, num=9))

    ax1.tick_params(axis='both', which='major',  labelsize=20)     
    ax2.tick_params(axis='both', which='major',  labelsize=20)

    title = 'S vs proba'
    ax1.set_title(title, fontsize=24, y=1.04)    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return   


def classifier_acc_comparison(df, basename):
    '''
    Fig. 2h, S14
        
        Parameters:
                    df:   pandas dataframe holding accuracy and auroc values
                          for the classifiers; should contains the following
                          columns: classifier, accuracy, auroc
    ''' 
    
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    y = -np.log10(1 - df['accuracy'].to_numpy())
    plt.bar(df['classifier'], y)

    y_min = 0.85
    y_max = 0.9973
    
    def t(x):
        return -np.log10(1 - x)
        
    ax.set_ylim(t(y_min), t(y_max))
    ytick = np.array([0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997])    
    ax.set_yticks(t(ytick))
    ax.set_yticklabels(ytick)


    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return  


def classifier_auroc_comparison(df, basename):
    '''
    Fig. 2h, S14
        
        Parameters:
                    df:   pandas dataframe holding accuracy and auroc values
                          for the classifiers; should contains the following
                          columns: classifier, accuracy, auroc
    ''' 
    
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    y = -np.log10(1 - df['auroc'].to_numpy())
    plt.bar(df['classifier'], y)


    y_min = 0.87
    y_max = 0.99955
    
    def t(x):
        return -np.log10(1 - x)
        
    ax.set_ylim(t(y_min), t(y_max))
    ytick = np.array([0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995])    
    ax.set_yticks(t(ytick))
    ax.set_yticklabels(ytick)


    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    return  

def peptide_yield_hbar(proba_bundle, basename):
    '''
    Fig. 3d
        
        Parameters:
          proba_bundle:   list of tuples 
                          (probabilities for a given sample, sample name)
    ''' 
    
    fig = plt.figure(figsize=(5, 10), dpi=300)
    ax = fig.add_subplot(111)
    
    from scipy.stats import sem

    means = list()
    errors = list()
    names = list()
    for sample in proba_bundle:
        means.append(np.mean(sample[0]))
        errors.append(sem(sample[0]))
        names.append(sample[1])

    means = means[::-1]
    errors = errors[::-1]
    names = names[::-1]  
    
    plt.barh(np.arange(len(means)), means, xerr=errors, 
             color='#323232', edgecolor='none', align='center',
             ecolor='#db5000')

    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels(names)

    ax.set_xlim(0, 1)
                                          
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()  


def feature_matrix(F, basename):
    '''
    Fig. 1c
        
        Parameters:
                    F:   feature matrix, 2D np.ndarray
    '''  
    
    fig = plt.figure(figsize=(16, 2), dpi=300)
    
    ax1 = plt.subplot2grid((1, 64), (0, 2), rowspan=1, colspan=62)
    ax2 = plt.subplot2grid((1, 64), (0, 0), rowspan=1, colspan=1)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    
    cmap = sns.light_palette("#3b61b1", as_cmap=True)
    ax1.imshow(F[::-1], cmap=cmap, norm=norm)
    
    #set ticks
    ax1.set_xticks(np.arange(0.5, F.shape[1] + 0.5, 20))
    ax1.set_xticklabels(np.arange(1, F.shape[1]+1, 20))
    ax1.set_xlabel('Fingerprint index', fontsize=20)
    ax1.yaxis.set_visible(False)
    
    
    ax2.set_yticks(np.arange(F.shape[0])+0.5)
    ax2.set_yticklabels(constants.aas[::-1])
    ax2.xaxis.set_visible(False)
    
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax1.set_title("F matrix", fontsize=22)
    
    #save png and svg, and close the file
    fig.savefig(basename + '.svg', bbox_inches = 'tight')
    fig.savefig(basename + '.png', bbox_inches = 'tight')
    plt.close()    

def Y_score(Y, basename):
    '''
    Fig. 2d, S13
        
        Parameters:
                    Y:   Y score matrix, 2D np.ndarray
    ''' 
    
    xTickNames = np.arange(1, Y.shape[1]+1, 1)
    yTickNames = constants.aas
    fig, ax = plt.subplots(1, 1, figsize=(9, 11.4), dpi=300)

    scale_min = -1.25
    scale_max = 1.25
    
    norm = mpl.colors.Normalize(vmin=scale_min, vmax=scale_max)
    c = ax.pcolor(Y, cmap=mpl.cm.cividis, norm=norm, edgecolors='w', linewidths=4)
    cbar = fig.colorbar(c, ax=ax)

    cbar.ax.set_ylabel("Y score", rotation=-90, va="bottom", fontsize=22)
    cbar.ax.tick_params(labelsize=20)    

    #set ticks
    ax.set_xticks(np.arange(Y.shape[1])+0.5)
    ax.set_yticks(np.arange(Y.shape[0])+0.5)
    ax.set_xticklabels(xTickNames)
    ax.set_yticklabels(yTickNames)

    #set labels
    ax.set_xlabel('Position within the random region', fontsize=25)
    ax.set_ylabel('Amino acid', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_title('Y-score map', fontsize=27)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()
    
def Y_score_var(var, basename):
    '''
    Fig. 2d, S13
        
        Parameters:
                    var:   Positional variance of Y scores, 1D np.ndarray
                           i.e., var = np.var(Y, axis=0)
    ''' 
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4), dpi=300)
    ax.bar(np.arange(var.size), var)
    
    #save png and svg, and close the file
    svg = basename + '.svg'
    png = basename + '.png'
    fig.savefig(svg, bbox_inches = 'tight')
    fig.savefig(png, bbox_inches = 'tight')
    plt.close()    