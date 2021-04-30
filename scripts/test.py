import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from numpy import random
from piecewise import piecewise
from scipy.stats import ks_2samp

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
with open("/Users/azhark/Documents/dev/SVMatrixGenerator/data/chr_sizes.txt") as f:
    for line in f:
        line = line.strip()
        (key, val) = line.split('\t')
        ranges[str(key)] = val


sample_files = ["~/iCloud/dev/SVMatrixGenerator/data/560_Breast/PD8660a2.560_breast.rearrangements.n560.bedpe.tsv"]
for file in sample_files:
    df = pd.read_csv(file, sep="\t") #bedpe file format: chrom1, start1, end1, chrom2, start2, end2, strand1, strand2, svclass(optional), sample
    if "strand1" in df.columns and "strand2" in df.columns:
        df = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "strand1", "strand2", "sample"]]
    df = df.astype({df.columns[1]: 'int32', df.columns[2]: 'int32', df.columns[5]: 'int32', df.columns[4]: 'int32', df.columns[0]: 'str', df.columns[3]: 'str'})
    sample = file.split('.')[0]
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

    df["svclass"] = svclass

    #GET SIZE
    sizes = [0 for x in svclass]
    for i, row in df.iterrows():
        if row["svclass"] != "trans":
            lengths.append(abs(row['start1'] - row['start2']))
            l = abs(row['start1'] - row['start2']) / 1000000 #megabases
            if l > 0.01 and l <= 0.1:
                size = "10-100kb"
                sizes[i] = size
            elif l > 0.1 and l <= 1:
                size = "100kb-1Mb"
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

    #num_events.append(df.shape[0] - 1) #number of rearrangements
    chrom_dfs = []
    all_chroms = set(list(df.chrom1.unique()) + list(df.chrom2.unique())) #all chromosomes with events on them

    for i, chromosome in enumerate(all_chroms): #apply PCF on a chromosome by chromosome basis
        df = df.filter(items=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'sample', 'svclass', 'size_bin', 'length'])
        chrom_df = df[(df.chrom1 == chromosome) | (df.chrom2 == chromosome)]

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

        #SIMULATION
        sim_df = chrom_df.copy()
        end = ranges[chromosome]
        for i, row in enumerate(sim_df.index):
            sim_df.iat[i, 1] = random.randint(end)
                #calculate inter-rearrangement distance
                coords = list(chrom_df[chrom_df.columns[1]])
                coords = sorted(coords)


        #calculate inter-rearrangement distance
        coords = list(chrom_df[chrom_df.columns[1]])
        coords = sorted(coords)

        sim_coords = list(sim_df[sim_df.columns[1]])
        sim_coords = sorted(sim_coords)
        chrom_inter_distances = []
        sim_inter_distances = []



        #defined as the number of base pairs from one rearrangement breakpoint to the one immediately preceding it in the reference genome
        if len(coords) > 2:
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

            sim_first_dist = sim_coords[1] - sim_coords[0]
            sim_last_dist = sim_coords[-1] - sim_coords[-2]

            #Repeat for simulated breakpoints
            for i in range(1, len(sim_coords)-1):
                j = i-1
                k = i+1
                if j >= 0 and k < len(sim_coords):
                    dist = min(sim_coords[i] - sim_coords[j], sim_coords[k] - sim_coords[i])
                    #assert(dist > 0)
                    sim_inter_distances.append(dist)
                if dist == 0:
                    print(sample)


            chrom_inter_distances = [first_dist] + chrom_inter_distances
            chrom_inter_distances.append(last_dist)

            sim_inter_distances = [sim_first_dist] + sim_inter_distances
            sim_inter_distances.append(sim_last_dist)

            ks_2samp(sim_df['length'], chrom_df["length"])
