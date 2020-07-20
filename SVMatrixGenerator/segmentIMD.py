import pandas as pd
import numpy as np
import pwlf
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
from numpy import random
#from GPyOpt.methods import BayesianOptimization

input_path = "/Users/khandekara2/iCloud/Alexandrov_Lab/data/SV/data/560_Breast/" #path to bedpe files
#input_path = "/Users/azhark/iCloud/Alexandrov_Lab/data/SV/data/Mutographs_ESCC_Train9/"
output_path = "/Users/khandekara2/iCloud/Alexandrov_Lab/data/SV/data/segments/560_Breast/" #path to segment files
#output_path = "/Users/azhark/iCloud/Alexandrov_Lab/data/SV/data/Mutographs_ESCC_Train9/segments/"
os.chdir(input_path)
#os.chdir("/Users/azhark/iCloud/Sherlock-Lung/")

#bedpe file format: chrom1, start1, end1, chrom2, start2, end2, strand1, strand2, svclass(optional), sample
#Given a list of input bedpe files corresponding to a cancer type (of the above format), calculate the inter-rearrangement distance for each rearrangement 
#and construct a plot (for each chromosome of each sample) with the location of the chromosome on the x-axis and the log IMD on the y-axis

#first navigate to directory containing files
sample_files = []
for file in os.listdir('.'):
    if file.endswith(".annot.bedpe.tsv") or file.endswith('.n560.bedpe.tsv'):
        sample_files.append(file)

sampleToSegments = {} #dictionary that maps (sample, chr, start) to its segmentAvgDensity and segmentAvgIMD

ranges = {} #key is chromosome, value is coordinate range
with open("chr_sizes.txt") as f:
    for line in f:
        line = line.strip()
        (key, val) = line.split('\t')
        ranges[str(key)] = val
    #print(ranges)


for file in sample_files:
    df = pd.read_csv(file, sep="\t") #bedpe file format: chrom1, start1, end1, chrom2, start2, end2, strand1, strand2, svclass(optional), sample
    df = df.astype({df.columns[1]: 'int32', df.columns[2]: 'int32', df.columns[5]: 'int32', df.columns[4]: 'int32', df.columns[0]: 'str', df.columns[3]: 'str'})
    sample = file.split('.')[0]
    #print (sample)
    
    sizes = [0 for x in list(df[df.columns[0]])]
    if "svclass" not in df.columns:
        #print("Determining SV Class")
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
             
                #GET SIZE
                if sv != "trans":
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
        
        df["svclass"] = svclass
    else: #SVCLASS column already present
        for i, row in df.iterrows(): #GET SIZE
            sv = row['svclass']
            if sv != "trans":
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
                size = ">10Mb"

    df["size_bin"] = sizes     

    #average IMD
    totalSampleIMD = 0
    sampleIMDCount = 0
    #average density
    totalSampleDensity = 0
    totalDensityCount = 0

    chrom_dfs = []
    
    all_chroms = set(list(df.chrom1.unique()) + list(df.chrom2.unique()))
    #print (all_chroms)
    for i, chromosome in enumerate(all_chroms): #apply PCF on a chromosome by chromosome basis
        if chromosome not in ranges.keys():
            print(chromosome)
            break

        df = df.filter(items=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'sample', 'svclass', 'size_bin'])
        #print(df.shape[0])
        chrom_df = df[(df.chrom1 == chromosome) | (df.chrom2 == chromosome)]
        if chrom_df.shape[0] <= 10:
            continue
        #print(chrom_df.shape[0])

        #keep track of partners
        d1 = dict(zip(list(chrom_df['start1']), list(chrom_df['start2'])))
        d2 = dict(zip(list(chrom_df['start2']), list(chrom_df['start1'])))
        d = {**d1, **d2} #combine dictionaries

        lb = chrom_df.iloc[:, 0:2] #get chrom1 and start1
        rb = chrom_df.iloc[:, 3:5] #get chrom2 and start2
        rest = chrom_df.iloc[:, 6:]
        
        lb = pd.DataFrame(np.concatenate((lb.values, rest.values), axis=1))
        rb = pd.DataFrame(np.concatenate((rb.values, rest.values), axis=1))
        lb.to_csv("lb.tsv", sep="\t", index=False)
        chrom_df.to_csv("test.tsv", sep="\t", index=False)
        
        lb.columns = ['chrom1', 'start1', 'sample', 'svclass', 'size_bin']
        rb.columns = ['chrom2', 'start2', 'sample', 'svclass', 'size_bin']
        
        #if either the right or left breakpoint is on this particular chromosome
        chrom = 1
        chr_lb = lb[lb.chrom1 == chromosome]
        chr_rb = rb[rb.chrom2 == chromosome]
        chrom_df = pd.DataFrame(np.concatenate((chr_lb.values, chr_rb.values), axis=0))
        
        chrom_df.columns = ['chrom', 'start', 'sample', 'svclass', 'size_bin']
        assert(chrom_df['chrom'].nunique() == 1)

        #sort on last column which is start coordinate
        chrom_df = chrom_df.sort_values(chrom_df.columns[1]) #CHROM, START

        #SIMULATION
        sim_df = chrom_df.copy()
        end = ranges[chromosome]
        for i, row in enumerate(sim_df.index):
            sim_df.iat[i, 1] = random.randint(end)

        #calculate inter-rearrangement distance
        coords = list(chrom_df[chrom_df.columns[1]])
        coords = sorted(coords)

        sim_coords = list(sim_df[sim_df.columns[1]])
        sim_coords = sorted(sim_coords)
        #print (coords)
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
                # if dist == 0 and j==i:
                #     print(coords[i] - coords[j])
                #     print(coords[k] - coords[i])
                    # print(coords)
                    # print(coords[i])
                    # print(coords[j])
                    # print(coords[j])
                    # print(sample)
                    # print(i)
                    # print(len(coords))
                    
                #assert(dist > 0)

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
                    # print(sim_coords[i] - sim_coords[j])
                    # print(sim_coords[k] - sim_coords[i])
                    # print(j,i,k)
                    # print(file)
                    # print(len(sim_coords))

            chrom_inter_distances = [first_dist] + chrom_inter_distances
            chrom_inter_distances.append(last_dist)

            sim_inter_distances = [sim_first_dist] + sim_inter_distances
            sim_inter_distances.append(sim_last_dist)

            #log inter-distances
            for i in range(len(chrom_inter_distances)):
                if chrom_inter_distances[i] != 0:
                    chrom_inter_distances[i] = math.log(chrom_inter_distances[i], 10)
            chrom_df['IMD'] = chrom_inter_distances

            for i in range(len(sim_inter_distances)):
                if sim_inter_distances[i] != 0:
                    sim_inter_distances[i] = math.log(sim_inter_distances[i], 10)
            sim_df['IMD'] = sim_inter_distances
            
            #chrom_df.to_csv("/Users/khandekara2/iCloud/com~apple~CloudDocs/Documents/Alexandrov_Lab/data/SV/data/segments/560_Breast/" + sample + "_" + chromosome + ".IMD.tsv", sep="\t", index=False)
            #chrom_df.to_csv("/Users/khandekara2/iCloud/Sherlock-Lung/segments/" + sample + "_" + chromosome + ".IMD.tsv", sep="\t", index=False)
            #sim_df.to_csv("/Users/khandekara2/iCloud/Sherlock-Lung/segments/" + sample + "_" + chromosome + ".IMD.tsv.simulated", sep="\t", index=False)
            chrom_df.to_csv(output_path + sample + "_" + chromosome + ".IMD.tsv", sep="\t", index=False)
            sim_df.to_csv(output_path + sample + "_" + chromosome + ".IMD.tsv.simulated", sep="\t", index=False)
print ("Outputted segment files to " + output_path)
    
