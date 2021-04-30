import pandas as pd
df1 = pd.read_csv("results/LDA/PanCNSigs_LDABayes_Signatures.tsv", sep="\t")
df2 = pd.read_csv("results/LDA/PanCNSigs_LDAGibbs_Signatures.tsv", sep="\t")
df3 = pd.read_csv("/Users/khandekara2/iCloud/dev/SVMatrixGenerator/data/PanCNSigs/signature_definitions.tsv", sep="\t")
channels = []
for c in bayes_signatures.index:
    channels.append(ldamodel.id2word[c])
bayes_signatures.index = channels
bayes_signatures = bayes_signatures.loc[classification, :]