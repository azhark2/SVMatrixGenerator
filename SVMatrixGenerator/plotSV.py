import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mplpatches
import matplotlib.ticker as ticker
from matplotlib.ticker import LinearLocator
import matplotlib.lines as lines
import matplotlib.transforms as transforms
import re
import os
import sys
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
import io

import string
import warnings
warnings.filterwarnings("ignore")

def plotSV(matrix_path, output_path, project, plot_type="pdf", percentage=False, aggregate=False):
    """Outputs a pdf containing Rearrangement signature plots

    :param matrix_path: path to matrix with 32 channels as rows and samples as columns
    :param output_path: path to output pdf file containing plots
    :param project: name of project
    :param plot_type: output type of plot (default:pdf)
    :param percentage: True if y-axis is displayed as percentage of CNV events, False if displayed as counts (default:False)
    :param aggregate: True if output is a single pdf of counts aggregated across samples(e.g for a given cancer type, y-axis will be counts per sample), False if output is a multi-page pdf of counts for each sample

    >>> plotSV()

    """

    #inner function to construct plot
    def plot(counts, labels, sample, project, percentage, aggregate=False):

        if percentage:
            counts = [(x/sum(counts))*100 for x in counts]

        color_mapping = {'del':{'>10Mb':"deeppink", '1Mb-10Mb':"hotpink", '10-100Kb':"lightpink", '100Kb-1Mb':"palevioletred", '1-10Kb':"lavenderblush"},
                     'tds':{'>10Mb':"saddlebrown", '1Mb-10Mb':"sienna", '10-100Kb':"sandybrown", '100Kb-1Mb':"peru", '1-10Kb':"linen"},
                     'inv':{'>10Mb':"rebeccapurple", '1Mb-10Mb':"blueviolet", '10-100Kb':"plum", '100Kb-1Mb':"mediumorchid", '1-10Kb':"thistle"}}

        alpha_dict = dict(enumerate(string.ascii_lowercase))
        x_labels = ['1-10kb', '10-100kb', '100kb-1Mb', '1Mb-10Mb','>10Mb']
        super_class = ['clustered', 'non-clustered']
        sub_class = ['del', 'tds', 'inv', 'trans']
        N=32
        ticks = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        width = 0.27
        xticks = []
        i = -1 #used to distinguish first bar from the rest
        fig, ax = plt.subplots(figsize=(16,8))
        
        # Custom Formatting
        plt.style.use('ggplot')
        plt.rcParams['axes.facecolor'] = 'white'
        plt.gca().yaxis.grid(True)
        plt.gca().grid(which='major', axis='y', color=[0.93,0.93,0.93], zorder=1)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(ticker.LinearLocator(5))
        ax.spines["bottom"].set_color("black")
        ax.spines["top"].set_color("black")
        ax.spines["right"].set_color("black")
        ax.spines["left"].set_color("black")
        plt.xlim(xmin=-.5,xmax=len(labels)-.5)
        tmp_max=max(counts)
        plt.ylim(ymax=1.25*tmp_max)
        # Add light gray horizontal lines at y-ticks
        ax.grid(linestyle='-', linewidth=1, color='#EDEDED', axis='y')

        for count, label in zip(counts, labels):
            categories = label.split('_')
            if len(categories) > 2:
                rearrangement_class = categories[1]
                size_class = categories[2]
            i += 1 #position of bar
            #print (categories)

            if len(categories) == 2: #clustered translocation or non-clustered translocation
                ax.bar(ticks[i], count, color="dimgray", edgecolor='black') #translocation only has one color
            else:
                ax.bar(ticks[i], count, color=color_mapping[rearrangement_class][size_class], edgecolor='black')

            xticks.append(ticks[i])
        ax.set_xticks(xticks)
        ax.set_xticklabels(x_labels * 3 + [' '] + x_labels * 3 + [' '], rotation=90, weight="bold", fontsize = 16, fontname='Arial', color='black')
        ax.tick_params(labelleft=True, left=False, bottom=False)
        ax.tick_params(axis='y', which='major', pad=0, labelsize=30)

        #ADD PATCHES AND TEXT
        patch_height = 0.05
        patch_width = 2.8
        loh_width= 2.5
        loh_len = 4.8

        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        
        #### CLUSTERED PATCHES ####
        ax.add_patch(plt.Rectangle((-.5, 1.095), 15.9, patch_height*1.5, clip_on=False, facecolor='gray', transform=trans))
        plt.text(6, 1.1125, "Clustered", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((-.5, 1.01), loh_len+.1, patch_height*1.5, clip_on=False, facecolor='maroon', transform=trans))
        plt.text(1.3, 1.03, "Del", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((4.6, 1.01), loh_len, patch_height*1.5, clip_on=False, facecolor='darkorange', transform=trans))
        plt.text(6.27, 1.03, "Tds", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((9.6, 1.01), loh_len, patch_height*1.5, clip_on=False, facecolor='slateblue', transform=trans))
        plt.text(11.35, 1.03, "Inv", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((14.6, 1.01), .8, patch_height*1.5, clip_on=False, facecolor='dimgray', transform=trans))
        plt.text(14.75, 1.03, "T", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        
        # add vertical black lines
        ax.axvline(x=15.5, color='black', linewidth=1)

        #### NON-CLUSTERED PATCHES ####
        ax.add_patch(plt.Rectangle((15.6, 1.095), 15.9, patch_height*1.5, clip_on=False, facecolor='black', transform=trans))
        plt.text(21, 1.1125, "Non-Clustered", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((15.6, 1.01), loh_len, patch_height*1.5, clip_on=False, facecolor='maroon', transform=trans))
        plt.text(17.35, 1.03, "Del", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((20.6, 1.01), loh_len, patch_height*1.5, clip_on=False, facecolor='darkorange', transform=trans))
        plt.text(22.25, 1.03, "Tds", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((25.6, 1.01), loh_len, patch_height*1.5, clip_on=False, facecolor='slateblue', transform=trans))
        plt.text(27.37, 1.03, "Inv", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)
        ax.add_patch(plt.Rectangle((30.6, 1.01), .9, patch_height*1.5, clip_on=False, facecolor='dimgray', transform=trans))
        plt.text(30.82, 1.03, "T", fontsize=23, fontname='Arial', fontweight='bold', color='white', transform=trans)

        
        # format the set_yticklabels labels
        if percentage:
            tmp_y_labels =['{0:0.1f}%'.format(round(x,1)) for x in ax.get_yticks().tolist()]
        else:
            tmp_y_labels =[round(x,1) for x in ax.get_yticks().tolist()]
        #ax.yaxis.labelpad = 300
            
        # set the y-axis labels
        ax.set_yticklabels(tmp_y_labels, fontname='Arial', weight='bold', fontsize=16, color='black')

        #y-axis titles
        if aggregate:
            ax.set_ylabel("# of events per sample", fontsize=24, fontname="Arial", weight = 'bold', labelpad = 15, color='black')
        elif percentage:
            ax.set_ylabel("Percentage(%)", fontsize=24, fontname="Arial", weight = 'bold', labelpad = 15, color='black')
            #ax.yaxis.labelpad = 1
        else:
            ax.set_ylabel("# of events", fontsize=24, fontname="Arial", weight = 'bold', labelpad = 15, color='black')

        #TITLE
        if not aggregate:
            plt.text(0, 0.90, sample, fontsize=20, fontname='Arial', fontweight='bold', color='black', transform=trans)
        else:
            plt.text(0, 0.90, project, fontsize=20, fontname='Arial', fontweight='bold', color='black', transform=trans)

        pp.savefig(fig, dpi=600, bbox_inches='tight')
    
    
    df = pd.read_csv(matrix_path, sep=None, engine='python') #flexible reading of tsv or csv
    label = df.columns[0]
    labels = df[label]
    if aggregate:
        num_samples = len(df.columns) - 1
        df['total_count'] = df.sum(axis=1) / num_samples #NORMALIZE BY # of SAMPLES
        counts = list(df['total_count'])
        sample = ''
        pp = PdfPages(output_path + project + '_RS32_counts_aggregated' + '.pdf')
        plot(counts, labels, sample, project, percentage, aggregate=True)
    else:
        if plot_type == 'pdf' and percentage:
            pp = PdfPages(output_path + project + '_RS32_signatures' + '.pdf')
        elif plot_type == 'pdf' and percentage==False:
            pp = PdfPages(output_path + project + '_RS32_counts' + '.pdf')
        else: #input == counts
            print("The only plot type supported at this time is pdf")

        #each column vector in dataframe contains counts for a specific sample
        samples = list(df)[1:]
        for i, (col, sample) in enumerate(zip(df.columns[1:], samples)):
            counts = list(df[col])
            counts = [(x/sum(counts))*100 for x in counts]
            assert(len(counts)) == 32
            assert(len(labels)) == 32
            plot(counts, labels, sample, project, percentage)
    pp.close()