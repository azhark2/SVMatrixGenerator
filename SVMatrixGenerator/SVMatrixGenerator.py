import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import random
from piecewise import piecewise
from scipy.stats import chisquare
import scipy.stats as stats
import seaborn as sns
import fastrand


# Rearrangement signatures. Clustered vs non-clustered rearrangements. We sought
# to separate rearrangements that occurred as focal catastrophic events or focal driver
# amplicons from genome-wide rearrangement mutagenesis using a piecewise constant fitting method. For each sample, both breakpoints of each rearrangement were
# considered individually and all breakpoints were ordered by chromosomal position.
# The inter-rearrangement distance, defined as the number of base pairs from one rearrangement breakpoint to the one immediately preceding it in the reference genome,
# was calculated. Putative regions of clustered rearrangements were identified as having
# an average inter-rearrangement distance that was at least 10 times greater than the
# whole-genome average for the individual sample. Piecewise constant fitting parameters used were γ=25 and kmin=10, with γ as the parameter that controls smoothness
# of segmentation, and kmin the minimum number of breakpoints in a segment.

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
with open("/Users/khandekara2/iCloud/dev/SVMatrixGenerator/data/chr_sizes.txt") as f:
    for line in f:
        line = line.strip()
        (key, val) = line.split('\t')
        ranges[str(key)] = val

#keep everything the same except the genomic location of SV event
def simulateBedpe(input_df, n, chromosome):
    sim_dfs = []
    end = int(ranges[chromosome])
    for i in range(n):
        sim_df = input_df.copy()
        for i, row in sim_df.iterrows():
            offset1 = row['end1'] - row['start1'] #this could be 0 or greater than 0 depending on the resolution of the caller
            offset2 = row['end2'] - row['start2'] #this could be 0 or greater than 0 depending on the resolution of the caller
            #new_coord = random.randint(end) #pick random start coordinate within chromosome
            new_coord = fastrand.pcg32bounded(end) #Fast random number generation in Python using PCG
            sim_df.iat[i, 1] = new_coord
            if (row["svclass"] != "trans"):
                if new_coord + row["length"] <= end: #if you don't go off end of chromosome
                    sim_df.iat[i, 1] = new_coord
                    sim_df.iat[i, 2] = new_coord + offset1
                    sim_df.iat[i, 4] = new_coord + row["length"]
                    sim_df.iat[i, 5] = new_coord + offset2
                else: #you go off end of chromosome
                    sim_df.iat[i, 4] = new_coord
                    sim_df.iat[i, 5] = new_coord + offset1
                    sim_df.iat[i, 1] = new_coord - row["length"]
                    sim_df.iat[i, 2] = new_coord + offset2
            else: #we are dealing with a translocation and the breakpoints are on another chromosome
                if sim_df[sim_df.columns[0]].nunique() == 1 and row["chrom1"] != row["chrom2"]:
                    sim_df.iat[i, 1] = new_coord
                    sim_df.iat[i, 2] = new_coord + offset1
                else:
                    sim_df.iat[i, 4] = new_coord
                    sim_df.iat[i, 5] = new_coord + offset1
        #recompute simulated lengths and make sure they are the same as actual
        lengths = []
        for i, row in sim_df.iterrows():
            lengths.append(abs(row['start1'] - row['start2']))
        sim_df["length"] == lengths

        assert(list(input_df["length"] == sim_df["length"])) #sizes of events are the same
        sim_dfs.append(sim_df)
    return sim_dfs

#distance in bp to nearest breakpoint that is not it's partner (not distance to breakpoint immediately preceding)
def computeIMD(chrom_df):
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
    lb.columns = ['chrom1', 'start1', 'sample', 'svclass', 'size_bin', "length"]
    rb.columns = ['chrom2', 'start2', 'sample', 'svclass', 'size_bin', "length"]

    chr_lb = lb[lb.chrom1 == chromosome]
    chr_rb = rb[rb.chrom2 == chromosome]
    chrom_df = pd.DataFrame(np.concatenate((chr_lb.values, chr_rb.values), axis=0))
    chrom_df.columns = ['chrom', 'start', 'sample', 'svclass', 'size_bin', "length"]
    assert(chrom_df['chrom'].nunique() == 1)

    #sort on last column which is start coordinate
    chrom_df = chrom_df.sort_values(chrom_df.columns[1]) #CHROM, START
    coords = list(chrom_df[chrom_df.columns[1]])
    coords = sorted(coords)

    chrom_inter_distances = []

    #defined as the number of base pairs from one rearrangement breakpoint to the one immediately preceding it in the reference genome

    for i in range(1, len(coords)-1):
        j = i-1
        k = i+1
        while j >= 0 and coords[j] == d[coords[i]]: #check if previous breakpoint is partner of this breakpoint, if it is, avoid it
            j=j-1
        while k < len(coords) and coords[k] == d[coords[i]]:
            k=k+1
        if j >= 0 and k < len(coords):
            dist = min(coords[i] - coords[j], coords[k] - coords[i])
        elif j < 0:
            dist = coords[k] - coords[i]
        else:
            dist = coords[i] - coords[j]
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

    # #INTERLEAVED VS NESTED CONFIGURATION
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
    if "strand1" in df.columns and "strand2" in df.columns:
        df = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "strand1", "strand2", "sample"]]
    df = df.astype({df.columns[1]: 'int32', df.columns[2]: 'int32', df.columns[4]: 'int32', df.columns[5]: 'int32', df.columns[0]: 'str', df.columns[3]: 'str'})


    #print(df.columns)
    lengths = []
    if "svclass" not in df.columns:
        if "strand1" not in df.columns or "strand2" not in df.columns:
            raise Exception("cannot classify rearrangements: svclass column missing, and cannot compute it because strand1 and strand2 are missing.")
        else:
            svclass = []
            for i, row in df.iterrows():
                if row['chrom1'] != row['chrom2']:
                    sv = "trans"
                    svclass.append("trans")
                elif row["strand1"] == '+' and row["strand2"] == '-' or row["strand1"] == '-' and row["strand2"] == '+':
                    sv = "inv"
                    svclass.append("inv")
                elif row["strand1"] == '+' and row["strand2"] == '+':
                    sv = "del"
                    svclass.append("del")
                elif row["strand1"] == '-' and row["strand2"] == '-':
                    sv = "tds"
                    svclass.append("tds")
                else:
                    raise Exception("cannot classify rearrangements: svclass column missing, and cannot compute it because strand1 and strand2 are not in the proper format.")
            print(svclass)
            df["svclass"] = svclass

    #GET SIZE
    sizes = [0 for x in svclass]
    #print(len(sizes))
    for i, row in df.iterrows():
        #print(i)
        if row["svclass"] != "trans":
            lengths.append(abs(row['start1'] - row['start2']))
            l = abs(row['start1'] - row['start2']) / 1000000 #megabases
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

    #CHECK FORMAT OF CHROMOSOME COLUMN ("chr1" vs. "1"), needs to be the latter
    if df['chrom1'][1].startswith("chr"):
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

    return df

#currently does not use copy number calls
#applies ShatterSeek statistical criteria
def detectChromothripsis(segment_df):
    equal_dist = False #EQUAL DISTRIBUTION of event classes
    total_events = segment_df.shape[0]
    events = set(["inv", "trans", "del", "tds"])
    event_counts = segment_df["svclass"].value_counts()

    for e in events: #in case any classes of events are missing entirely
        if e not in event_counts.keys():
            event_counts[e] = 0
    expected = [round(total_events / 4) for x in events]
    observed = np.array(list(event_counts.values))
    # if expected[0] <= 5: #fishers exact test for small # of observations
    #     oddsratio, pvalue = stats.fisher_exact([observed], [expected])
    # else: #chi-squared test
    #     t, pvalue = chisquare(observed, expected)
    t, pvalue = chisquare(observed, expected)
    equal_dist_p.append(pvalue)
    if pvalue > 0.05:
        equal_dist = True

    #determine what proportion of events are in a interleaved configuration
    if 'nested' in segment_df["Configuration"].value_counts() and 'interleaved' in segment_df["Configuration"].value_counts():
        frac_interleaved = segment_df["Configuration"].value_counts()['interleaved'] / (segment_df["Configuration"].value_counts()['interleaved'] + segment_df["Configuration"].value_counts()['nested'])
    elif 'nested' in segment_df["Configuration"].value_counts() and not 'interleaved' in segment_df["Configuration"].value_counts():
        frac_interleaved = 0
    else:
        frac_interleaved = 1

    interleaved_frac.append(frac_interleaved)
    if frac_interleaved >=interleaved_cutoff and equal_dist: #equal distribution of events, majority interleaved, + clustered = chromothripsis
        annotation = ['clustered:C' for i in range(segment_df.shape[0])]
        #print("These clustered events can likely be attributed to chromothripsis")
        #print("The fraction of interleaved events was " + str(frac_interleaved) + " and the probability of events being drawn from an equal distribution is " + str(pvalue))
    else:
        annotation = ['clustered:NC' for i in range(segment_df.shape[0])]

    segment_df['Annotation'] = annotation
    return segment_df

#compare two rearrangement matrices(produced by different tools for example)
def compareInputMatrix(df1, df2):
    cos_sim = []
    df2 = df2[df1.columns] #make sure columns are same order so samples match up
    for a1, w1 in zip(df1.columns[1:], df2.columns[1:]):
        a = np.array(df1[a1])
        b = np.array(df2[w1])
        dot = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        cos = dot / (norma * normb)
        cos_sim.append(cos)
    return cos_sim

#construct rainfall plot of SV's on chromosome
def plotIMD(imd, sim_imd, sample, chromosome, output_path):
    #sizes  = {"0": 60, '>10Mb':np.float16(50), '1Mb-10Mb':np.float16(40), '1-10Kb':np.float16(10), '100kb-1Mb':np.float16(30), '10-100kb':np.float16(20)}
    sizes  = {"0":60, '>10Mb':50, '1Mb-10Mb':40, '1-10Kb':10, '100kb-1Mb':30, '10-100kb':20}
    size_order = ['1-10Kb', '10-100kb', '100kb-1Mb', '1Mb-10Mb', '>10Mb', "0"]
    #imd.columns = ['chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', 'log IMD']
    #print(imd.dtypes)
    #sim_imd = pd.read_csv(file + '.simulated', sep="\t")
    #sim_imd.columns = ['chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', 'log IMD']

    #log IMD for plotting purposes
    if imd.shape[0] > 0:
        pp = PdfPages(output_path + sample + '_IMD_plots' + '.pdf')
        log_imd = [np.log(float(x)) for x in np.array(imd["IMD"])]
        imd['log IMD'] = log_imd
        log_imd = [np.log(float(x)) for x in np.array(sim_imd["IMD"])]
        sim_imd['log IMD'] = log_imd
        #imd.columns = ['chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', 'log IMD']
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        #print(imd.dtypes)
        imd = imd.astype({"start": int, "length":int})
        a = sns.scatterplot("start", "log IMD", data=imd, hue="svclass", size="size_bin", sizes=sizes, size_order=size_order, legend="full", ax=axes[0]).set_title(sample + ": " + "chr" + str(chromosome))
        b = sns.scatterplot("start", "log IMD", data=sim_imd, hue="svclass", size="size_bin", sizes=sizes, size_order=size_order, legend="full", ax=axes[1]).set_title(sample + ": " + "chr" + str(chromosome) + ' (Simulated)')
        #plt.savefig(sample + "_" + chromosome +".png", dpi=150)
        try:
            pp.savefig(fig, dpi=150, bbox_inches='tight')
        except ValueError:  #raised if `y` is empty.
            print(file)
        pp.close()
    print("Rainfall IMD plots created for " + sample)
    print ("Saved IMD plots to " + output_path + sample + '_IMD_plots' + '.pdf')

#currently converts annotated bedpe to old SV32 classification
def tsv2matrix(original_df, annotated_df):
    samples = annotated_df["sample"].unique()
    features = ['clustered_del_1-10Kb', 'clustered_del_10-100Kb', 'clustered_del_100Kb-1Mb', 'clustered_del_1Mb-10Mb', 'clustered_del_>10Mb', 'clustered_tds_1-10Kb', 'clustered_tds_10-100Kb', 'clustered_tds_100Kb-1Mb', 'clustered_tds_1Mb-10Mb', 'clustered_tds_>10Mb', 'clustered_inv_1-10Kb', 'clustered_inv_10-100Kb', 'clustered_inv_100Kb-1Mb', 'clustered_inv_1Mb-10Mb', 'clustered_inv_>10Mb', 'clustered_trans', 'non-clustered_del_1-10Kb', 'non-clustered_del_10-100Kb', 'non-clustered_del_100Kb-1Mb', 'non-clustered_del_1Mb-10Mb', 'non-clustered_del_>10Mb', 'non-clustered_tds_1-10Kb', 'non-clustered_tds_10-100Kb', 'non-clustered_tds_100Kb-1Mb', 'non-clustered_tds_1Mb-10Mb', 'non-clustered_tds_>10Mb', 'non-clustered_inv_1-10Kb', 'non-clustered_inv_10-100Kb', 'non-clustered_inv_100Kb-1Mb', 'non-clustered_inv_1Mb-10Mb', 'non-clustered_inv_>10Mb', 'non-clustered_trans']

    # with open('/Users/khandekara2/iCloud/dev/SVMatrixGenerator/RS32_features.txt') as f:
    #     next(f)
    #     for line in f:
    #         features.append(line.strip())
    arr = np.zeros((32, len(samples)), dtype='int')
    nmf_matrix = pd.DataFrame(arr, index=features, columns=samples)
    print()

    #record the classification for all the individual breakpoints(clustered, non-clustered, etc.)
    breakpointToAnnot = {}
    for i, row in annotated_df.iterrows():
        breakpointToAnnot[(row["sample"], row['chrom'], row['start'])] = row["Annotation"]

    #go through original bedpe, look up annotation, and fill in matrix
    for i, row in original_df.iterrows():
        b1 = (row["sample"], row['chrom1'], row['start1'])
        b2 = (row["sample"], row['chrom2'], row['start2'])
        channel1 = ''
        channel2 = ''
        if b1 in breakpointToAnnot:
            if breakpointToAnnot[b1] == "clustered:NC" or  breakpointToAnnot[b1] == 'clustered:C':
                if row["svclass"] != "trans": #size has to be taken into account
                    channel1 = "clustered_" + row["svclass"] + "_" + row["size_bin"]
                else:
                    channel1 = "clustered_" + row["svclass"]
            else:
                if row["svclass"] != "trans":
                    channel1 = "non-clustered_" + row["svclass"] + "_" + row["size_bin"]
                else:
                    channel1 = "non-clustered_" + row["svclass"]
        if b2 in breakpointToAnnot:
            if breakpointToAnnot[b2] == "clustered:NC" or  breakpointToAnnot[b2] == 'clustered:C':
                if row["svclass"] != "trans": #size has to be taken into account
                    channel2 = "clustered_" + row["svclass"] + "_" + row["size_bin"]
                else:
                    channel2 = "clustered_" + row["svclass"]
            else:
                if row["svclass"] != "trans":
                    channel2 = "non-clustered_" + row["svclass"] + "_" + row["size_bin"]
                else:
                    channel2 = "non-clustered_" + row["svclass"]
        else: #if the event is not annotated, than assume that it is non-clustered
            if row["svclass"] != "trans":
                channel = "non-clustered_" + row["svclass"] + "_" + row["size_bin"]
            else:
                channel = "non-clustered_" + row["svclass"]

        if channel1.split("_")[0] == "clustered":
            channel = channel1
        if channel2.split("_")[0] == "clustered":
            channel = channel2
        nmf_matrix.at[channel, row["sample"]] += 1

    nmf_matrix.index.name = 'Mutation Types'
    nmf_matrix.reindex([features])
    return nmf_matrix

counts = {} #master dictionary that stores sample: channel
#sample_files = ["~/iCloud/dev/SVMatrixGenerator/data/560_Breast/PD8660a2.560_breast.rearrangements.n560.bedpe.tsv"]

#PCAWG MELANOMA
# sample_files = []
# for file in os.listdir("/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/SV/MELA-AU/"):
#     if file.endswith(".bedpe.tsv"):
#         sample_files.append(file)

#MESOTHELIA SAMPLES
# sample_files = []
# for file in os.listdir('/Users/khandekara2/iCloud/dev/Mesothelia-Tumors/data/SV_MESO_release_22012020/'):
#     if file.endswith(".consensus.sur.bedpe.tsv"):
#         sample_files.append(file)

#560 BREAST SAMPLES
# sample_files = []
# for file in os.listdir('/Users/khandekara2/iCloud/dev/HRD-Signatures/data/560_Breast/'):
#     if file.endswith(".n560.bedpe.tsv"):
#         sample_files.append(file)

#os.chdir('/Users/azhark/iCloud/dev/HRD-Signatures/data/560_Breast/')
os.chdir("/Users/khandekara2/iCloud/dev/Mesothelia-Tumors/data")
#os.chdir("/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/SV/MELA-AU")
#os.chdir('/Users/khandekara2/iCloud/dev/Mesothelia-Tumors/data/SV_MESO_release_22012020/')

num_events = []
clustered_p = []
num_segments = []
equal_dist_p = []
interleaved_frac = []

#KEY PARAMETERS
clustered_cutoff = 0.01 #fraction of cases from simulations where we see an average IMD equal to or less than the observed IMD
interleaved_cutoff = 0.75 #fraction of events that have to be in the interleaved configuration in order to count as chromothripsis
num_simulations = 100 #of simulations

#PCF PARAMETERS (these parameters affect the segmentation)

output_path = "/Users/khandekara2/iCloud/dev/SVMatrixGenerator/results/"
#sample_files = ["/Users/khandekara2/iCloud/dev/SVMatrixGenerator/data/Melanoma_chromo_DO220896.bedpe.tsv"]
#input is a list of bedpe files for a given cohort of samples

project = "MELA-AU"
#project = "560-Breast"
samples = set([])
all_segments = [] #all segment df's with complete annotation
#process bedpe
#file = "/Users/khandekara2/iCloud/dev/Mesothelia-Tumors/data/SV/MESO.all.bedpe"
#file = "/Users/azhark/iCloud/dev/HRD-Signatures/data/560_Breast/560_breast.rearrangements.bedpe"
file = "/Users/khandekara2/iCloud/dev/SVMatrixGenerator/data/PCAWG/SV/MELA-AU_SV.bedpe.tsv"
data = pd.read_csv(file, sep="\t") #bedpe file format: chrom1, start1, end1, chrom2, start2, end2, strand1, strand2, svclass(optional), sample
print("Creating structural variant matrix for the " + str(data["sample"].nunique()) + " samples in " + project)
with open(output_path + project + 'SV32.log', 'w') as f: #log file which contains information about results
    for sample in data["sample"].unique():
        #print(sample)
        df = data[data["sample"] == sample]
        df = df.reset_index()
        samples.add(sample)
        #track samples

        df = processBEDPE(df)

        all_chroms = set(list(df.chrom1.unique()) + list(df.chrom2.unique())) #all chromosomes with events on them

        for i, chromosome in enumerate(all_chroms): #apply PCF on a chromosome by chromosome basis
            chrom_df = df[(df.chrom1 == chromosome) | (df.chrom2 == chromosome)]
            if chrom_df.shape[0] > 3 and chromosome != 'Y' and chromosome != "X":
                num_events.append(chrom_df.shape[0])
                chrom_df.reset_index(drop=True, inplace=True)
                imd_df = computeIMD(chrom_df) #table with all breakpoints considered individually and IMD calculated
                #print("Starting simulations")
                sim_dfs = simulateBedpe(chrom_df, num_simulations, chromosome)
                for i in range(len(sim_dfs)):
                    sim_dfs[i] = computeIMD(sim_dfs[i])
                #plotIMD(imd_df, sim_dfs[0], "DO220896", chromosome, output_path) #creates rainfall plot pdf for each chromosome for given sample

                null_dist = [] #null distribution of average inter-mutational distances derived from simulations
                for sim_df in sim_dfs:
                    null_dist.append(np.mean(sim_df['IMD']))
                null_dist = np.array(null_dist)
                #print("Finished running simulations")

                #plt.hist(null_dist)
                #APPLY PCF
                #print("Applying PCF")
                x = list(imd_df.start)
                y = list(imd_df.IMD)
                model = piecewise(x, y)
                #print("Finished applying PCF")

                for i in range(len(model.segments)):
                    #print("Analyzing segment " + str(i))
                    start = int(model.segments[i].start_t)
                    end = int(model.segments[i].end_t)
                    segment_df = imd_df[(imd_df['start'] >= start) & (imd_df['start'] <= end)]
                    segAvgIMD = np.mean(np.array(segment_df["IMD"])) #mean IMD in this segment is our measure of interest
                    assert(len(null_dist > 0))
                    p = np.sum(null_dist <= segAvgIMD) / len(null_dist)
                    clustered_p.append(p) #fraction of cases where simulated mean IMD is equal to or lower than observed average IMD in the segment
                    #print("The probability that these " + str(segment_df.shape[0]) + " events on chromosome " + str(chromosome) + " in segment # " + str(i) + " are clustered by chance is " + str(p))
                    #clustered_p.append(p)

                    if p <= clustered_cutoff:
                        #CLUSTERED, so apply additional chromothripsis criteria
                        #print("There is clustering on chromosome " + str(chromosome) + " between " + str(start) + " and " + str(end))
                        #print("Now applying additional chromothripsis criteria")
                        segment_df = detectChromothripsis(segment_df) #this table will already have events annotated as clustered:NC or clustered:C
                    else:
                        annotation = ["non-clustered" for i in range(segment_df.shape[0])]
                        segment_df['Annotation'] = annotation
                    all_segments.append(segment_df)
            else: #either too few events on chromosomes or we are dealing with sex chromosomes
                annotation = ["non-clustered" for i in range(chrom_df.shape[0])]
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


result_df = pd.concat(all_segments) #concatenate all individual segment dataframes into one large result dataframe

#save annotated dataframe for given project
result_df.to_csv("/Users/khandekara2/iCloud/dev/SVMatrixGenerator/results/" + project + ".annotated.bedpe.tsv", sep="\t", index=False)

matrix = tsv2matrix(df, result_df) #matrix for NMF
matrix.to_csv("/Users/khandekara2/iCloud/dev/SVMatrixGenerator/results/" + project + ".matrix.tsv", sep="\t", index=False)



