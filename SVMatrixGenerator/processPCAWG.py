import os
import pandas as pd
import csv
os.chdir("/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/raw_data")

df = pd.read_csv('/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/PCAWG_SV_METADATA.tsv', sep="\t")
file_names = [x.split(".")[0] for x in df['File Name']]
df['Sample'] = file_names              

sampleToProject = dict(zip(df["File Name"], df["Project"]))

with open("/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/PCAWG_SV.bedpe.tsv", 'w') as csvout:
    writer = csv.writer(csvout, delimiter="\t")
    header = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "strand1", "strand2", "sample", "project", "donor", "specimen"]
    writer.writerow(header)
    for file in os.listdir("."):
        if file.endswith(".somatic.sv.bedpe"):
            sample = file.split(".")[0]
            with open("/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PCAWG/raw_data/" + file, 'r') as f:
                next(f)
                sample_df = df[df["Sample"] == sample]
                for line in f:
                    info = [list(sample_df["Sample ID"].unique())[0], list(sample_df["Project"].unique())[0], list(sample_df["ICGC Donor"].unique())[0], list(sample_df["Specimen Type"].unique())[0]]
                    row = line.strip().split("\t")
                    writer.writerow(row[:6] + [row[8], row[9]] + [info[0]] + info[1:])


