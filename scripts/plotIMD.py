import seaborn as sns
import pandas as pd
import numpy as np
import pwlf
import math
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#         chrom   start   sample  svclass size_bin        IMD
# 7       7365075 NSLC-0057-T01   inversion       1-10Kb  0.0
# 7       7369129 NSLC-0057-T01   inversion       1-10Kb  0.0

#input_dir = "/Users/khandekara2/iCloud/Sherlock-Lung/segments/"
input_dir = "/Users/azhark/iCloud/dev/Pediatric-Tumors/results/IMD-Plots/"
#input_dir = "/Users/khandekara2/iCloud/Alexandrov_Lab/data/SV/data/Mutographs_ESCC_Train9/segments/"
os.chdir(input_dir)


project = 'KiCS'
#project = 'Mutographs-ESCC'
output_path = "/Users/azhark/iCloud/dev/Pediatric-Tumors/results/IMD-Plots/"
pp = PdfPages(output_path + project + '_IMD_plots' + '.pdf')
#pp = PdfPages(project + '_IMD_plots' + '.pdf')

sizes  = {'>10Mb':50, '1Mb-10Mb':40, '1-10Kb':10, '100kb-1Mb':30, '10-100kb':20}
size_order = ['1-10Kb', '10-100kb', '100kb-1Mb', '1Mb-10Mb', '>10Mb']

count = 0
for file in os.listdir('.'):
    if file.endswith(".IMD.tsv"):
        sample = file.split(".")[0].split("_")[0]
        c = file.split(".")[0].split("_")[-1]
        imd = pd.read_csv(file, sep="\t")
        if imd.shape[0] > 50:
            
            #print(file)
            imd.columns = ['chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', 'log IMD']
            #print(imd.dtypes)
            sim_imd = pd.read_csv(file + '.simulated', sep="\t")
            sim_imd.columns = ['chrom', 'Genomic Coordinate', 'sample', 'svclass', 'size_bin', 'log IMD']
            if imd.shape[0] > 0:
                fig, axes = plt.subplots(1, 2, figsize=(20, 6))
                a = sns.scatterplot("Genomic Coordinate", "log IMD", data=imd, hue="svclass", size="size_bin", sizes=sizes, size_order=size_order, legend="full", ax=axes[0]).set_title(sample + ": " + "chr" + str(c))
				
				
				
                b = sns.scatterplot("Genomic Coordinate", "log IMD", data=sim_imd, hue="svclass", size="size_bin", sizes=sizes, size_order=size_order, legend="full", ax=axes[1]).set_title(sample + ": " + "chr" + str(c) + ' (Simulated)')
                plt.savefig(sample + "_" + c +".png", dpi=150)
                try:
                    pp.savefig(fig, dpi=150, bbox_inches='tight')
                    count+=1
                except ValueError:  #raised if `y` is empty.
                    print(file)
                    pass
                
print(count)            
            
pp.close()
print("There were %s sample IMD plots created" %count)        
print ("Saved IMD plots to " + output_path + project + '_IMD_plots' + '.pdf')
