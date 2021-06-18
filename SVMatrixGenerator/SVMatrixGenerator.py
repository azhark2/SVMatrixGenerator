import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import math
import os
import matplotlib.pyplot as plt
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

from numpy import random
from piecewise import piecewise
from piecewise import piecewise_plot
from scipy.stats import chisquare
import scipy.stats as stats
import seaborn as sns
import fastrand
import pickle
from statistics import mode

# Rearrangement signatures. Clustered vs non-clustered rearrangements. We sought
# to separate rearrangements that occurred as focal catastrophic events or focal driver
# amplicons from genome-wide rearrangement mutagenesis using a piecewise constant fitting method. For each sample, both breakpoints of each rearrangement were
# considered individually and all breakpoints were ordered by chromosomal position.

# The inter-rearrangement distance, defined as the number of base pairs from one rearrangement breakpoint to the one immediately preceding it in the reference genome,
# was calculated.

#Putative regions of clustered rearrangements were identified as having
# an average inter-rearrangement distance that was at least 10 times greater than the
# whole-genome average for the individual sample

#Piecewise constant fitting parameters used were γ=25 and kmin=10, with γ as the parameter that controls smoothness
# of segmentation, and kmin the minimum number of breakpoints in a segment

# chrom1    start1  end1    chrom2  start2  end2    sv_id   pe_support  strand1 strand2 svclass svmethod
# X 31906363    31906364    X   33495419    33495420    SVMERGE1    50  +   -   DEL SNOWMAN_BRASS

#' The column "svclass" should correspond to (Sanger BRASS convention): (strand1/strand2)
#'
#' inversion (+/-), if mates on the same chromosome
#'
#' inversion (-/+), if mates on the same chromosome
#'
#' deletion (+/+), if mates on the same chromosome
#'
#' tandem-duplication (-/-), if mates on the same chromosome
#'
#' translocation, if mates are on different chromosomes

#BREAKPOINT DISTANCE
#sample.rearrs$bkdist <- abs(sample.rearrs$start2 - sample.rearrs$start1)

#rearrs.left <- sv_bedpe[,c('chrom1','start1','sample')]
#rearrs.right <- sv_bedpe[,c('chrom2','start2','sample')]

#EXAMPLE WITH SV CLASS PRESENT
# chrom1    start1  end1    chrom2  start2  end2    sv_id   pe_support  strand1 strand2 svclass svmethod
# 19    21268384    21268385    19  21327858    21327859    SVMERGE6    9   +   -   DEL SNOWMAN_DELLY

#EXAMPLE WITHOUT SV CLASS PRESENT
# chrom1  start1  end1    chrom2  start2  end2    sample  strand1 strand2
# 1       84379707        84379707        1       84862686        84862686        PD9847a -       -

#DEFINE CHROMOSOME BOUNDARIES
ranges = {} #key is chromosome, value is coordinate range
with open("/Users/azhark/iCloud/dev/SVMatrixGenerator/data/chr_sizes.txt") as f:
    for line in f:
        line = line.strip()
        (key, val) = line.split('\t')
        ranges[str(key)] = val

# def getMedianIMD():
#     df = pd.read_csv("", sep="\t")
#     df = processBEDPE(df)
#     computeIMD()

#keep everything the same except the genomic location of SV event
def simulateBedpe(input_df, n, chromosome):
    """This is the summary line

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.
    """

    sim_dfs = []
    end = int(ranges[str(chromosome)])
    #end = abs(np.max(np.array(input_df.start1)) -  np.min(np.array(input_df.start1)))
    # if input_df.shape[0] <= 3 or end <= 3:
    #   return [input_df]
    for i in range(n):
        sim_df = input_df.copy()
        r=-1
        for row in input_df.itertuples():
            r = r+1
            offset1 = int(row.end1) - int(row.start1) #this could be 0 or greater than 0 depending on the resolution of the caller
            offset2 = int(row.end2) - int(row.start2) #this could be 0 or greater than 0 depending on the resolution of the caller
            #new_coord = random.randint(end) #pick random start coordinate within chromosome

            #once you have the new start coordinate, where to put the end coordinate so that resolution and length is preserved?
            new_coord = end+1
            while new_coord + row.length >= end or new_coord + offset1 + row.length >= end:
                new_coord = fastrand.pcg32bounded(end) #Fast random number generation in Python using PCG
            sim_df.iat[r, 1] = new_coord
            sim_df.iat[r, 2] = new_coord + offset1
            sim_df.iat[r, 4] = new_coord + row.length
            sim_df.iat[r, 5] = new_coord + offset2

        lengths = []
        for row in sim_df.itertuples():
            lengths.append(abs(row.start1 - row.start2))
        sim_df["length"] == lengths

        assert(list(input_df["length"] == sim_df["length"])) #sizes of events are the same
        assert(all(i > 0 for i in list(sim_df['start1'])))
        assert(all(i > 0 for i in list(sim_df['start2'])))
        sim_dfs.append(sim_df)
    return sim_dfs

#distance in bp to nearest breakpoint that is not it's partner (not distance to breakpoint immediately preceding)
def computeIMD(chrom_df, chromosome):

    

    #keep track of partners
    d1 = dict(zip(list(chrom_df['start1']), list(chrom_df['start2'])))
    d2 = dict(zip(list(chrom_df['start2']), list(chrom_df['start1'])))
    d = {**d1, **d2} #combine dictionaries

    lb = chrom_df.iloc[:, 0:2] #get chrom1 and start1
    rb = chrom_df.iloc[:, 3:5] #get chrom2 and start2
    rest = chrom_df.iloc[:, 6:]

    lb = pd.DataFrame(np.concatenate((lb.values, rest.values), axis=1))
    rb = pd.DataFrame(np.concatenate((rb.values, rest.values), axis=1))

    #BREAKPOINTS ARE CONSIDERED INDIVIDUALLY

    #['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'sample', 'svclass', 'size_bin', 'length']
    lb.columns = ['chrom1', 'start1', 'sample', 'svclass', 'size_bin', "length"]
    rb.columns = ['chrom2', 'start2', 'sample', 'svclass', 'size_bin', "length"]

    chr_lb = lb[lb.chrom1 == chromosome]
    chr_rb = rb[rb.chrom2 == chromosome]
    # print(chr_lb)
    # print(chr_rb)
    chrom_df = pd.DataFrame(np.concatenate((chr_lb.values, chr_rb.values), axis=0))
    chrom_df.columns = ['chrom', 'start', 'sample', 'svclass', 'size_bin', "length"]

    # print(chrom_df['chrom'].unique())
    #assert(chrom_df['chrom'].nunique() == 1)

    #sort on last column which is start coordinate
    chrom_df = chrom_df.sort_values(chrom_df.columns[1]) #CHROM, START
    coords = list(chrom_df[chrom_df.columns[1]])
    coords = sorted(coords)

    chrom_inter_distances = []

    #defined as the number of base pairs from one rearrangement breakpoint to the one closest to it that is not it's partner
    for i in range(1, len(coords)-1):
        j = i-1
        k = i+1
        while j >= 0 and coords[j] == d[coords[i]]: #check if previous breakpoint is partner of this breakpoint, if it is, avoid it
            j=j-1
        while k < len(coords) and coords[k] == d[coords[i]]:
            k=k+1
        if j >= 0 and k < len(coords):
            if coords[i] - coords[j] == 0:
                dist = coords[k] - coords[i]
            elif coords[k] - coords[i] == 0:
                dist = coords[i] - coords[j]
            else:
                dist = min(coords[i] - coords[j], coords[k] - coords[i])
        elif j < 0:
            dist = coords[k] - coords[i]
        else:
            dist = coords[i] - coords[j]

        if dist == 0:
            print(coords[j], coords[i], coords[k])
            print(len(coords))
        chrom_inter_distances.append(dist)

    #now we take care of the edge cases of the first and last breakpoint

    if coords[1] == d[coords[0]]:
        first_dist = coords[2] - coords[0]
    else:
        first_dist = coords[1] - coords[0]

    if coords[-2] == d[coords[-1]]:
        last_dist = coords[-1] - coords[-3]
    else:
        last_dist = coords[-1] - coords[-2]

    chrom_inter_distances = [first_dist] + chrom_inter_distances
    chrom_inter_distances.append(last_dist)
    chrom_df['IMD'] = chrom_inter_distances

    #INTERLEAVED VS NESTED CONFIGURATION
    configuration = ['interleaved' for i in range(len(coords))]
    for i in range(1, len(coords)):
        j = i-1
        while coords[j] == d[coords[i]] and not (d[coords[i]] < max(d[coords[j]], coords[j]) and coords[i] < max(d[coords[j]], coords[j]) and d[coords[i]] > min(d[coords[j]], coords[j]) and coords[i] > min(d[coords[j]], coords[j])): #check if previous breakpoint is partner of this breakpoint, if it is, avoid it
            j=j-1
        if j >= 0: #determine if we have a nested or interleaved configuration
            if d[coords[i]] < max(d[coords[j]], coords[j]) and coords[i] < max(d[coords[j]], coords[j]) and d[coords[i]] > min(d[coords[j]], coords[j]) and coords[i] > min(d[coords[j]], coords[j]):
                configuration[i] = "nested"

    chrom_df["Configuration"] = configuration
    return chrom_df

#reformat input bedpe files
def processBEDPE(df):
    """A simple function that process a given bedpe file produced by an SV caller"""
    
    #CHECK FORMAT OF CHROMOSOME COLUMN ("chr1" vs. "1"), needs to be the latter
    if df['chrom1'][0].startswith("chr"):
        chrom1 = []
        chrom2 = []
        for a, b in zip(df['chrom1'], df['chrom2']):
            if a.startswith("chr") or b.startswith("chr"):
                a = a[3:]
                b = b[3:]
                chrom1.append(a)
                chrom2.append(b)
            else:
                break

        df['chrom1'] = chrom1
        df['chrom2'] = chrom2

    df = df[(df["chrom1"] != 'X') & (df["chrom1"] != 'Y') & (df["chrom2"] != 'X') & (df["chrom2"] != 'Y')]

    if "strand1" in df.columns and "strand2" in df.columns:
        df = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "strand1", "strand2", "sample"]]
    else:
        df = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "sample", "svclass"]]
    df = df.astype({df.columns[1]: 'int32', df.columns[2]: 'int32', df.columns[4]: 'int32', df.columns[5]: 'int32', df.columns[0]: 'str', df.columns[3]: 'str'})

    lengths = []
    if "svclass" not in df.columns:
        if "strand1" not in df.columns or "strand2" not in df.columns:
            raise Exception("cannot classify rearrangements: svclass column missing, and cannot compute it because strand1 and strand2 are missing.")
        else:
            svclass = []
            for row in df.itertuples():
                if row.chrom1 != row.chrom2:
                    sv = "translocation"
                    svclass.append(sv)
                elif row.strand1 == '+' and row.strand2 == '-' or row.strand1 == '-' and row.strand2 == '+':
                    sv = "inversion"
                    svclass.append(sv)
                elif row.strand1 == '+' and row.strand2 == '+':
                    sv = "deletion"
                    svclass.append(sv)
                elif row.strand1 == '-' and row.strand2 == '-':
                    sv = "tandem-duplication"
                    svclass.append(sv)
                else:
                    raise Exception("cannot classify rearrangements: svclass column missing, and cannot compute it because strand1 and strand2 are not in the proper format.")
            #f.write(svclass)
            df["svclass"] = svclass
    else:
        svclass = list(df["svclass"])

    #GET SIZE
    sizes = [0 for x in svclass]
    i=-1
    for row in df.itertuples():
        i=i+1
        if row.svclass != "translocation":
            lengths.append(abs(row.start1 - row.start2))
            l = abs(row.start1 - row.start2) / 1000000 #megabases
            if l > 0.01 and l <= 0.1:
                size = "10-100Kb"
                sizes[i] = size
            elif l > 0.1 and l <= 1:
                size = "100Kb-1Mb"
                sizes[i] = size
            elif l > 1 and l <= 10:
                size = "1Mb-10Mb"
                sizes[i] = size
            elif l > 0.001 and l <= 0.010:
                size = "1-10Kb"
                sizes[i] = size
            else:
                size = ">10Mb"
                sizes[i] = size
        else:
            sizes[i] = "0"
            lengths.append(0)

    df["size_bin"] = sizes
    df["length"] = lengths
    df = df.filter(items=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'sample', 'svclass', 'size_bin', 'length'])

    return df

#currently does not use copy number calls
#applies ShatterSeek statistical criteria
def detectChromothripsis(segment_df):
    """This is the summary line

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.
    """

    equal_dist = False #EQUAL DISTRIBUTION of event classes
    total_events = segment_df.shape[0]
    #print(total_events)
    events = set(["inversion", "translocation", "deletion", "tandem-duplication"])
    event_counts = segment_df["svclass"].value_counts()

    for e in events: #in case any classes of events are missing entirely
        if e not in event_counts.keys():
            event_counts[e] = 0
    #print(event_counts)
    expected = [round(total_events / 4) for x in events]
    observed = np.array(list(event_counts.values))
    #print(observed)
    # if expected[0] <= 5: #fishers exact test for small # of observations
    #     oddsratio, pvalue = stats.fisher_exact([observed], [expected])
    # else: #chi-squared test
    #     t, pvalue = chisquare(observed, expected)
    t, pvalue = chisquare(observed, expected)
    equal_dist_p.append(pvalue)

    #record the p value for equal distribution of events
    equal_dist_pval = []
    for i in range(segment_df.shape[0]):
        equal_dist_pval.append(pvalue)

    if pvalue >= 0.05:
        equal_dist = True

    #determine what proportion of events are in a interleaved configuration
    if 'nested' in segment_df["Configuration"].value_counts() and 'interleaved' in segment_df["Configuration"].value_counts():
        frac_interleaved = segment_df["Configuration"].value_counts()['interleaved'] / (segment_df["Configuration"].value_counts()['interleaved'] + segment_df["Configuration"].value_counts()['nested'])
    elif 'nested' in segment_df["Configuration"].value_counts() and not 'interleaved' in segment_df["Configuration"].value_counts():
        frac_interleaved = 0
    else:
        frac_interleaved = 1

    #print(frac_interleaved)

    #record the fraction of events on this chromosome
    interleaved_frac = []
    for i in range(segment_df.shape[0]):
        interleaved_frac.append(frac_interleaved)
    #print(interleaved_frac)

    if frac_interleaved >=interleaved_cutoff and equal_dist: #equal distribution of events, majority interleaved, + clustered = chromothripsis
        annotation = ['clustered:C' for i in range(segment_df.shape[0])]
        #f.write("These clustered events can likely be attributed to chromothripsis")
        #f.write("The fraction of interleaved events was " + str(frac_interleaved) + " and the probability of events being drawn from an equal distribution is " + str(pvalue))
    else:
        annotation = ['clustered:NC' for i in range(segment_df.shape[0])]

    segment_df['Annotation'] = annotation
    segment_df["interleaved_frac"] = interleaved_frac
    segment_df["equal_dist_pval"] = equal_dist_pval
    return segment_df

#construct rainfall plot of SV's on chromosome
def plotIMD(sampleToEvents, output_path): #sampleToEvents contains sample: (imd_df, sim_df, model)
    """This is the summary line

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.
    """

    sizes  = {"0":60, '>10Mb':50, '1Mb-10Mb':40, '1-10Kb':10, '100Kb-1Mb':30, '10-100Kb':20}
    size_order = ['1-10Kb', '10-100Kb', '100Kb-1Mb', '1Mb-10Mb', '>10Mb', "0"]

    #log IMD for plotting purposes
    for k,v in sampleToEvents.items(): #(sample: (list of annotated dataframes for each chromosome, list of simulated dataframes, pcf model object))
        sample = k
        annotated_dfs = v[0] #list of each chromosome's annotated df required for plotting
        sim_imds = v[1] #list of each chromosome's simulated df required for plotting
        plotting_models = v[2] #list of each chromosome's pcf model object containing segmentations
        pp = PdfPages(output_path + sample + '_IMD_plots' + '.pdf') #multi-page pdf, one for each chromosome
        for imd, sim_imd, model in zip(annotated_dfs, sim_imds, plotting_models): #for each chromsome for this particular sample
            labels=[]
            chromosome = mode(list(imd["chrom"]))
            if imd.shape[0] > 0: #it's possible a chromosome has no rearrangements
            #take the log of the IMD for plotting purposes only
                # log_imd = [np.log(float(x)) for x in np.array(imd["IMD"])]
                # imd['log IMD'] = log_imd
                # log_imd = [np.log(float(x)) for x in np.array(sim_imd["IMD"])]
                # sim_imd['log IMD'] = log_imd

                #imd.columns = ['chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', 'log IMD']
                fig, axes = plt.subplots(1, 2, figsize=(20, 6))
                #f.write(imd.dtypes)

                imd = imd.astype({"start": int, "length":int})
                markers = {'clustered:C': "X", "non-clustered": "s", 'clustered:NC':'*'}
                a = sns.scatterplot("start", "IMD", data=imd, hue="svclass", size="size_bin", sizes=sizes, size_order=size_order, legend="full", style="Annotation", markers=markers, ax=axes[0]).set_title(sample + ": " + "chr" + str(chromosome))

                #print('Num segments: %s' % len(model.segments))
                #plt.plot(t, v, '.', alpha=0.6)
                for i, seg in enumerate(model.segments):
                    t_new = [seg.start_t, seg.end_t]
                    v_hat = [seg.predict(t) for t in t_new]
                    axes[0].plot(t_new, v_hat, 'k-')
                #print(sim_imd)
                if sim_imd.shape[0] > 0: #dataframe will be empty if there are less than 6 events, so no plotting of simulations
                    b = sns.scatterplot("start", "IMD", data=sim_imd, hue="svclass", size="size_bin", sizes=sizes, size_order=size_order, legend="full", ax=axes[1]).set_title(sample + ": " + "chr" + str(chromosome) + ' (Simulated)')
                # print(labels)
            pp.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close()
            #plt.savefig(sample + "_" + chromosome +".png", dpi=150)
        pp.close()

#currently converts annotated bedpe to old SV32 classification
def tsv2matrix(original_df, annotated_df):
    """This is the summary line

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.
    """
    samples = annotated_df["sample"].unique()
    features = ['clustered_del_1-10Kb', 'clustered_del_10-100Kb', 'clustered_del_100Kb-1Mb', 'clustered_del_1Mb-10Mb', 'clustered_del_>10Mb', 'clustered_tds_1-10Kb', 'clustered_tds_10-100Kb', 'clustered_tds_100Kb-1Mb', 'clustered_tds_1Mb-10Mb', 'clustered_tds_>10Mb', 'clustered_inv_1-10Kb', 'clustered_inv_10-100Kb', 'clustered_inv_100Kb-1Mb', 'clustered_inv_1Mb-10Mb', 'clustered_inv_>10Mb', 'clustered_trans', 'non-clustered_del_1-10Kb', 'non-clustered_del_10-100Kb', 'non-clustered_del_100Kb-1Mb', 'non-clustered_del_1Mb-10Mb', 'non-clustered_del_>10Mb', 'non-clustered_tds_1-10Kb', 'non-clustered_tds_10-100Kb', 'non-clustered_tds_100Kb-1Mb', 'non-clustered_tds_1Mb-10Mb', 'non-clustered_tds_>10Mb', 'non-clustered_inv_1-10Kb', 'non-clustered_inv_10-100Kb', 'non-clustered_inv_100Kb-1Mb', 'non-clustered_inv_1Mb-10Mb', 'non-clustered_inv_>10Mb', 'non-clustered_trans']

    arr = np.zeros((32, len(samples)), dtype='int')
    nmf_matrix = pd.DataFrame(arr, index=features, columns=samples)
    nmf_matrix.info()
    #record the classification for all the individual breakpoints(clustered, non-clustered, etc.)
    breakpointToAnnot = {}
    for row in annotated_df.itertuples():
        breakpointToAnnot[(row.sample, row.chrom, row.start)] = row.Annotation

    svclass_mapping = {"deletion":"del", "tandem-duplication":"tds", "inversion":"inv", "translocation":"trans"}
    svclass2 = [svclass_mapping[x] for x in original_df.svclass]
    original_df.svclass=svclass2

    #go through original bedpe, look up annotation, and fill in matrix
    for row in original_df.itertuples():
        channel1 = ''
        channel2 = ''
        b1 = (row.sample, row.chrom1, row.start1)
        b2 = (row.sample, row.chrom2, row.start2)

        if b1 in breakpointToAnnot:
            if breakpointToAnnot[b1] == "clustered:NC" or  breakpointToAnnot[b1] == 'clustered:C':
                if row.svclass != "trans": #size has to be taken into account
                    channel1 = "clustered_" + row.svclass + "_" + row.size_bin
                else:
                    channel1 = "clustered_" + row.svclass
            else: #
                if row.svclass != "trans":
                    channel = "non-clustered_" + row.svclass + "_" + row.size_bin
                else:
                    channel = "non-clustered_" + row.svclass
        if b2 in breakpointToAnnot:
            if breakpointToAnnot[b2] == "clustered:NC" or  breakpointToAnnot[b2] == 'clustered:C':
                if row.svclass != "trans": #size has to be taken into account
                    channel2 = "clustered_" + row.svclass + "_" + row.size_bin
                else:
                    channel = "clustered_" + row.svclass
            else:
                if row.svclass != "trans":
                    channel = "non-clustered_" + row.svclass + "_" + row.size_bin
                else:
                    channel2 = "non-clustered_" + row.svclass
        if b1 not in breakpointToAnnot and b2 not in breakpointToAnnot: #if the event is not annotated, than assume that it is non-clustered
            if row.svclass != "trans":
                channel = "non-clustered_" + row.svclass + "_" + row.size_bin
            else:
                channel = "non-clustered_" + row.svclass

        #if either breakpoint in the pair of breakpints is annotated as clustered, the enitre event is considered clustered
        if channel1 != '':
            channel=channel1
        if channel2 != '':
            channel=channel2

        if channel1.split("_")[0] == "clustered":
            channel = channel1
        if channel2.split("_")[0] == "clustered":
            channel = channel2
        nmf_matrix.at[channel, row.sample] += 1

    nmf_matrix.index.name = 'Mutation Types'
    nmf_matrix.reindex([features]).reset_index()

    return nmf_matrix

num_events = []
clustered_p = []
num_segments = []
equal_dist_p = []

sampleToEvents = {} #contains the annotated bedpe for each sample for plotting purposes
def annotateBedpe(project, file, output_path):
    """This is the summary line

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.
    """
    samples = set([])
    all_segments = [] #all segment df's with complete annotation
    sim_dist = [] #distribution of average IMD from all simulations
    observed_dist = []

    data = pd.read_csv(file, sep="\t") #bedpe file format: chrom1, start1, end1, chrom2, start2, end2, strand1, strand2, svclass(optional), sample
    data = processBEDPE(data)
    print("Creating structural variant matrix for the " + str(data["sample"].nunique()) + " samples in " + project)
    with open(output_path + project + '.SV32.log', 'w') as f: #log file which contains information about results
        for sample in data["sample"].unique():
            
            plotting_dfs = []
            sim_plotting_dfs = []
            plotting_models = []
            f.write("STARTING ANALYSIS FOR " + str(sample) + "..." + "/n")
            df = data[data["sample"] == sample]
            #df = df.reset_index()
            samples.add(sample)
            #print(df)

            all_chroms = set(list(df.chrom1.unique()) + list(df.chrom2.unique())) #all chromosomes with events on them

            for i, chromosome in enumerate(all_chroms): #apply PCF on a chromosome by chromosome basis
                sample_segments = []
                #f.write("Analyzing SV's on for " + str(sample) + "on chromosome " + str(chromosome))
                chrom_df = df[(df.chrom1 == chromosome) | (df.chrom2 == chromosome)]

                if chrom_df.shape[0] >= 10:
                    num_events.append(chrom_df.shape[0])
                    chrom_df.reset_index(drop=True, inplace=True)

                    imd_df = computeIMD(chrom_df, chromosome)
                    imd_df = imd_df.sort_values(by=['start'], ascending=True) #table with all breakpoints considered individually and IMD calculated
                    f.write("Starting simulations")

                    #produce n simulated dataframes where everything in the input data is kept the same except for the genomic location of the SV
                    sim_dfs = simulateBedpe(chrom_df, num_simulations, chromosome)
                    for i in range(len(sim_dfs)):
                        sim_dfs[i] = computeIMD(sim_dfs[i], chromosome)
                        sim_df = sim_dfs[i]
                        x = list(sim_df.start)
                        y = list(sim_df.IMD)
                        sim_model = piecewise(x, y)
                        #print(len(model.segments))
                        for i in range(len(sim_model.segments)):
                            start = int(sim_model.segments[i].start_t)
                            end = int(sim_model.segments[i].end_t)
                            segment_df = sim_df[(sim_df['start'] >= start) & (sim_df['start'] <= end)]

                            log_imd = []
                            for v in np.array(sim_df["IMD"]):
                                if v > 0:
                                    log_imd.append(math.log10(v))  
                                else:
                                    log_imd.append(0)
                            sim_df["log IMD"] = log_imd

                            null_dist.append(np.mean(sim_df['log IMD']))


                    #f.write("Finished running simulations")
                    #plt.hist(null_dist)
                    #APPLY PCF
                    #f.write("Applying PCF")
                    x = list(imd_df.start)
                    y = list(imd_df.IMD)
                    model = piecewise(x, y)

                    #plotIMD(imd_df, sim_dfs[0], sample, chromosome, "/Users/azhark/iCloud/dev/SVMatrixGenerator/plots/", model) #creates rainfall plot pdf for each chromosome for given sample

                    #f.write("Finished applying PCF")
                    # p = piecewise_plot(x, y, model=model)
                    # p.savefig("/Users/azhark/iCloud/dev/SVMatrixGenerator/plots/pcf/" + sample + chromosome + ".png", dpi=200)
                    for i in range(len(model.segments)):
                        f.write("Analyzing segment " + str(i))
                        start = int(model.segments[i].start_t)
                        end = int(model.segments[i].end_t)
                        segment_df = imd_df[(imd_df['start'] >= start) & (imd_df['start'] <= end)]
                        if segment_df.shape[0] >= min_seg_events:
                            segment_df = segment_df.sort_values(by=['start'], ascending=True)
            #               # print(sample, chromosome, str(i+1), start, end)
            #               # print(segment_df)
                            # sim_dfs = simulateBedpe(chrom_df[(chrom_df['start1'] >= start) & (chrom_df['start1'] <= end)], num_simulations, chromosome)
                            # for i in range(len(sim_dfs)):
                            #   sim_dfs[i] = computeIMD(sim_dfs[i], chromosome)
            #                 sim_dist.append(np.mean(np.array(sim_dfs[i]["IMD"])))
                            null_dist = [] #null distribution of average inter-mutational distances derived from simulations
                            for sim_df in sim_dfs:

                                log_imd = []
                                for v in np.array(sim_df["IMD"]):
                                    if v > 0:
                                        log_imd.append(math.log10(v))  
                                    else:
                                        log_imd.append(0)
                                sim_df["log IMD"] = log_imd

                                null_dist.append(np.mean(sim_df['log IMD']))
                                
                            null_dist = np.array(null_dist)


                            log_imd = []
                            for v in np.array(segment_df["IMD"]):
                                if v > 0:
                                    log_imd.append(math.log10(v))  
                                else:
                                    log_imd.append(0)

                            segment_df["log IMD"] = log_imd

                            segAvgIMD = np.mean(np.array(segment_df["log IMD"])) #mean IMD in this segment is our measure of interest
                            assert(len(null_dist > 0))
                            p_clustered = np.sum(null_dist <= segAvgIMD) / len(null_dist)
                            clustered_p.append(p_clustered) #fraction of cases where simulated mean IMD is equal to or lower than observed average IMD in the segment
                            #f.write("The probability that these " + str(segment_df.shape[0]) + " events on chromosome " + str(chromosome) + " in segment # " + str(i) + " are clustered by chance is " + str(p_clustered))
                        else: #make p-value really large so everything is considered unclustered
                            p_clustered = p_clustered + clustered_cutoff 
            #           #clustered_p.append(p)

                        if p_clustered <= clustered_cutoff: #simulation derived p-value
                            #CLUSTERED, so apply additional chromothripsis criteria
                            f.write("There is clustering on chromosome " + str(chromosome) + " between " + str(start) + " and " + str(end))
                            f.write("Now applying additional chromothripsis criteria")
                            segment_df = detectChromothripsis(segment_df) #this table will already have events annotated as clustered:NC or clustered:C
                        else:
                            annotation = ["non-clustered" for i in range(segment_df.shape[0])]
                            segment_df['Annotation'] = annotation
                        all_segments.append(segment_df)
                        sample_segments.append(segment_df)
                         #actual annotated df, simulated df, pcf model
                    sim_imd = sim_dfs[0]

                else: #either too few events on chromosomes or we are dealing with sex chromosomes
                    annotation = ["non-clustered" for i in range(chrom_df.shape[0])]

                    #simulate just for plotting purposes
                    if chrom_df.shape[0] > 10:
                        sim_dfs = simulateBedpe(chrom_df, 1, chromosome) 
                        sim_imd = computeIMD(sim_dfs[0], chromosome)
                    else:
                        sim_imd = pd.DataFrame()

                    imd = [0 for i in range(chrom_df.shape[0])]
                    config = ["interleaved" for i in range(chrom_df.shape[0])]
                    start = list(chrom_df["start1"])
                    chrom = list(chrom_df["chrom1"])

                    chrom_df["Annotation"] = annotation
                    chrom_df["Configuration"] = config
                    chrom_df["IMD"] = imd
                    chrom_df["chrom"] = chrom
                    chrom_df["start"] = start
                    chrom_df = chrom_df[["chrom", "start", "sample", "svclass", "size_bin", "length", "IMD", "Configuration", "Annotation"]]
                    all_segments.append(chrom_df)
                    sample_segments.append(chrom_df)
                    model = '' #for plotting function
                # print(chromosome)
                # print(sample_segments)
                df2 = pd.concat(sample_segments) #annotated df for chromosome
                #print(df2)

                plotting_dfs.append(df2)
                sim_plotting_dfs.append(sim_imd)
                plotting_models.append(model)
                # print(sim_imd)
                # print(model)
            
            sampleToEvents[sample] = (plotting_dfs, sim_plotting_dfs, plotting_models)
            
    pickle.dump(sampleToEvents, open("/Users/azhark/iCloud/dev/SVMatrixGenerator/plots/" + project + "_sampleToEvents.pickle", 'wb'))
    #plotIMD(sampleToEvents, "/Users/azhark/iCloud/dev/SVMatrixGenerator/plots/")
    # rainfallPlot = plotIMD(sampleToEvents, "/Users/azhark/iCloud/dev/SVMatrixGenerator/plots/") #creates rainfall plot pdf for each chromosome for given sample
    # rainfallPlot.savefig(project + "_" + sample + '_IMD_plots' + '.pdf', dpi=300)
    # print("Rainfall IMD plots created for " + sample)
    # print("Saved IMD plots to " + output_path + project + "_" + sample + '_IMD_plots' + '.pdf')
    result_df = pd.concat(all_segments)
    return result_df, data

#PCF PARAMETERS (these parameters affect the segmentation)
clustered_cutoff = 0.01 #fraction of cases from simulations where we see an average IMD equal to or less than the observed IMD
interleaved_cutoff = 0.75 #fraction of events that have to be in the interleaved configuration in order to count as chromothripsis
num_simulations = 100 #of simulations
min_seg_events = 4 #minimum amount of breakpoints in the pcf segment

file = "/Users/azhark/iCloud/dev/Mesothelia-Tumors/data/MESO.all.bedpe"
#file = "/Users/azhark/iCloud/dev/HRD-Signatures/data/560_Breast/560_breast.rearrangements.bedpe"
#file = "/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/SV/MELA-AU_SV.bedpe.tsv"
project = "Mesothelia-Tumors"
output_path = "/Users/azhark/iCloud/dev/SVMatrixGenerator/results/"

if __name__ == "__main__":

    result_df, original_df = annotateBedpe(project, file, output_path)
    #save annotated dataframe for given sample
    result_df.to_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/" + project + ".annotated.bedpe.tsv", sep="\t", index=False)
    matrix = tsv2matrix(original_df, result_df) #matrix for NMF
    matrix.reset_index().to_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/" + project + ".matrix.tsv", sep="\t", index=False)

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
            #f.write (categories)

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

        plt.show()
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
            f.write("The only plot type supported at this time is pdf")

        #each column vector in dataframe contains counts for a specific sample
        samples = list(df)[1:]
        for i, (col, sample) in enumerate(zip(df.columns[1:], samples)):
            counts = list(df[col])
            counts = [(x/sum(counts))*100 for x in counts]
            assert(len(counts)) == 32
            plot(counts, labels, sample, project, percentage)
    pp.close()

#matrix.to_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/MELA-AU_PCAWG.matrix.tsv", sep="\t", index=False)

#matrix.to_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/560-Breast.matrix.tsv", sep="\t", index=False)

#compare two rearrangement matrices(produced by different tools for example)
def compareInputMatrix(df1, df2):
    """This is the summary line

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.
    """
    cos_sim = []
    df1 = df1.iloc[:, 1:]
    df2 = df2.iloc[:, 1:]
    df2 = df2[df1.columns] #make sure columns are same order so samples match up
    for a1, w1 in zip(df1.columns, df2.columns):
        a = np.array(df1[a1])
        b = np.array(df2[w1])
        dot = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        cos = dot / (norma * normb)
        cos_sim.append(cos)
    return cos_sim

#COSINE SIMILARITY HISTOGRAMS
# mela1 = pd.read_csv("/Users/azhark/iCloud/dev/Mesothelia-Tumors/data/MELA-AU.rearrangements.bedpe.matrix.tsv", sep="\t")
# mela2 = pd.read_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/MELA-AU_PCAWG.matrix.tsv", sep="\t")
# plt.hist(melanoma)

# meso1 = pd.read_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/Mesothelia.rearrangements.bedpe.matrix.tsv", sep="\t")
# meso2 = pd.read_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/Mesothelia.rearrangements.bedpe.STL.matrix.tsv", sep="\t")
# meso = compareInputMatrix(meso1, meso2)
#f.write(meso2.shape)
# plt.hist(meso, bins=10, align='left', color='b', edgecolor='red',
#               linewidth=1)
#
# matrix.reset_index().head()
# #SigProfiler
# project = "PCAWG-MELA-AU"
# matrix_path = "/Users/azhark/iCloud/dev/SVMatrixGenerator/results/" + project + ".matrix.tsv"
# output_path = "/Users/azhark/iCloud/dev/SVMatrixGenerator/results/"
# plotSV(matrix_path, output_path, project, plot_type="pdf", percentage=False, aggregate=True)

#SignatureToolsLib
# matrix_path = "/Users/azhark/iCloud/dev/Mesothelia-Tumors/data/Mesothelia.rearrangements.bedpe.STL.matrix.tsv"
# output_path = "/Users/azhark/iCloud/dev/SVMatrixGenerator/results/"
# project = "Mesothelia_SV32_Signature.Tools.Lib"
# plotSV(matrix_path, output_path, project, plot_type="pdf", percentage=False, aggregate=True)


#DIAGNOSTICS:
# null_dist.hist()
# plt.hist(clustered_p)
#
# result_df = pd.read_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/results/Mesothelia.annotated.bedpe.tsv", sep="\t")
# svclass_counts = result_df.Annotation.value_counts()
# plt.figure(figsize=(3,2))
# sns.barplot(svclass_counts.index, svclass_counts.values, alpha=0.8)
# plt.title('Mesothelia Classifciation')
# plt.ylabel('Frequency', fontsize=12)
# plt.xlabel('Annotation', fontsize=12)
# plt.xticks(
#     rotation=45,
#     horizontalalignment='right',
#     fontweight='light',
#     fontsize='small'
# )
#
# plt.show()
# actual_df = pd.read_csv("/Users/azhark/iCloud/dev/Mesothelia-Tumors/data/Mesothelia.rearrangements.bedpe.STL.matrix.tsv", sep="\t")
# c = actual_df.iloc[0:16, :]
# nc = actual_df.iloc[16:, :]
# nc.iloc[:, 1:].to_numpy().sum()

# for c in actual_df.iloc[:, 0]:
#     if

#'chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', "length", 'log IMD'

# new_sizes = []
# for s in result_df["size_bin"]:
#     if s == "100kb-1Mb":
#         new_sizes.append("100Kb-1Mb")
#     elif s == "10-100kb":
#         new_sizes.append("10-100Kb")
#     else:
#         new_sizes.append(s)
# result_df["size_bin"] = new_sizes
#
# new_sizes = []
# for s in df["size_bin"]:
#     if s == "100kb-1Mb":
#         new_sizes.append("100Kb-1Mb")
#     elif s == "10-100kb":
#         new_sizes.append("10-100Kb")
#     else:
#         new_sizes.append(s)
# df["size_bin"] = new_sizes




# pcawg = pd.read_csv("/Users/azhark/iCloud/dev/SVMatrixGenerator/PCAWG_Chromothripsis_Calls.tsv", sep="\t")
# pcawg.info()
