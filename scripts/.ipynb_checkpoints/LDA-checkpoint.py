import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
import string
from gensim.models.wrappers import LdaMallet
import pickle
from gensim.models.hdpmodel import HdpModel
from gensim.models.ldamulticore import LdaMulticore

alpha_dict = dict(enumerate(string.ascii_uppercase))
mallet_path = "/Users/azhark/iCloud/bin/mallet-2.0.8/bin/mallet"
output_path = "/Users/azhark/iCloud/dev/SVMatrixGenerator/results/LDA/"

def matrixToBag(matrix_path):
    label_set = set([])
    df = pd.read_csv(matrix_path, sep="\t")
    mutations = [] #list of vectors, each vector represents a sample
    labels = list(df[df.columns[0]])
    assert(len(labels) == 48)
    for i, col in enumerate(df.columns[1:]): #for every sample
        counts = list(df[col])
        assert(len(counts) == 48)
        sample = []
        for j, c in enumerate(counts):
#             if labels[j] == '9+:LOH:>40Mb':
#                 print(j, c)
            if c > 0:
                for k in range(c):
                    sample.append(labels[j])
                    label_set.add(labels[j])

        mutations.append(sample)

    idToChannel = corpora.Dictionary(mutations) #maps id to channel name
    bagOfMutations = [idToChannel.doc2bow(mutation) for mutation in mutations] #each tuple contains (id(0 to 47 corresponding to 48 features), frequency)

    return bagOfMutations, idToChannel

def get_activities(model):

    activities = pd.DataFrame(np.zeros((len(model[bagOfMutations]), len(model[bagOfMutations[0]]))))

    columns = []
    for i in range(activities.shape[1]):
        columns.append("Signature " + alpha_dict[i])
    activities.columns = columns

    for i in range(len(activities)):
        for j, col in enumerate(activities.columns):
            activities.at[i, col] = model[bagOfMutations][i][j][1] #[(0, 0.0071448083), (1, 0.97140515), (2, 0.0071500205), (3, 0.007152412), (4, 0.007147603)]
            #print(i, j, col)

    return activities

#make sample by feature matrix for nmf with rows as features and samples as columns
classification = []
with open('/Users/azhark/iCloud/dev/SVMatrixGenerator/data/CNV_features.tsv') as f:
    for line in f:
        classification.append(str(line.strip("\n")))

paths = ["/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PanCNSigs/PanCNSigs_TCGA.matrix.tsv"] 

paths = ["/Users/azhark/iCloud/dev/SVMatrixGenerator/data/PanPCAWG_CNV48.matrix.tsv", "/Users/khandekara2/iCloud/dev/counts/560-Breast/560-Breast.CNV.matrix.tsv", '/Users/azhark/iCloud/Alexandrov_Lab/matrices/Mutographs_ESCC_Battenberg_high_confidence.matrix.tsv', "~/iCloud/dev/Pediatric-Tumors/data/KiCS200_ASCAT_NGS.CNV.matrix.tsv"]

#paths = paths1 + paths2 + paths3
projectToSigs = {"PanCNSigs_TCGA":19, "560-Breast":6, "Sherlock_Lung_All_Segments":5, "Mutographs_ESCC_Battenberg_high_confidence":4, "KiCS200_ASCAT_NGS":6, 'PanPCAWG': 19, "PanPCAWG_CNV48":19}

#loop through input matrices of cancer types and perform LDA
for path in paths:
    project = path.split("/")[-1].split(".")[0]
    print("Executing LDA algorithm on " + project)
    num_sigs = projectToSigs[project]
    bagOfMutations, idToChannel = matrixToBag(path)

    ldamodel = LdaMulticore(bagOfMutations, num_topics=num_sigs, id2word = idToChannel, passes=100, iterations=100, minimum_probability=0)
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bagOfMutations, num_topics=num_sigs, id2word=idToChannel, iterations=100, topic_threshold=0.0)
    #hdpmodel = HdpModel(bagOfMutations, idToChannel, K=20, T=48)


    pickle.dump(ldamodel, open(output_path + project + '_lda_model.pickle', 'wb'))
    pickle.dump(ldamallet, open(output_path + project + '_mallet_model.pickle', 'wb'))
    #pickle.dump(hdpmodel, output_path + project + '_hdp_model.pickle')

    bayes_signatures = pd.DataFrame(ldamodel.get_topics().transpose())
    columns = []
    for i in range(bayes_signatures.shape[1]):
        columns.append("Signature " + alpha_dict[i])
    bayes_signatures.columns = columns
    channels = []
    for c in bayes_signatures.index:
        channels.append(ldamodel.id2word[c])
		
	if len(channels) != 48: #one channel had 0 counts for all samples in dataset
		channel_set = set(channels)
		for c in classification:
			if c not in channel_set:
				channels.append(c)
				bayes_signatures.loc[len(bayes_signatures)] = 0
		
    bayes_signatures.index = channels
    bayes_signatures = bayes_signatures.loc[classification, :]

    gibbs_signatures = pd.DataFrame(ldamodel.get_topics().transpose())
    columns = []
    for i in range(gibbs_signatures.shape[1]):
        columns.append("Signature " + alpha_dict[i])
    gibbs_signatures.columns = columns

    # hdp_signatures = pd.DataFrame(hdpmodel.get_topics().transpose())
    # print(hdp_signatures.shape)
    # columns = []
    # for i in range(hdp_signatures.shape[1]):
    #     columns.append("Signature " + alpha_dict[i])
    # hdp_signatures.columns = columns
    # channels = []
    # for c in hdp_signatures.index:
    #     channels.append(hdpmodel.id2word[c])
    # hdp_signatures.index = channels
    # hdp_signatures = hdp_signatures.loc[classification, :]

    channels = []
    for c in gibbs_signatures.index:
        channels.append(ldamodel.id2word[c])
		
	if len(channels) != 48: #one channel had 0 counts for all samples in dataset
		channel_set = set(channels)
		for c in classification:
			if c not in channel_set:
				channels.append(c)
				gibbs_signatures.loc[len(gibbs_signatures)] = 0
	
	assert(channels == 48)
				
				
		
    gibbs_signatures.index = channels
	
	gibbs_signatures = gibbs_signatures.loc[classification, :]

   

    bayes_signatures.reset_index(inplace=True)
    bayes_signatures.to_csv(output_path + project + "_LDABayes_Signatures.tsv", sep="\t", index=False)

    gibbs_signatures.reset_index(inplace=True)
    gibbs_signatures.to_csv(output_path + project + "_LDAGibbs_Signatures.tsv", sep="\t", index=False)

    # hdp_signatures.reset_index()
    # hdp_signatures.to_csv(output_path + project + "_HDP_Signatures.tsv", sep="\t", index=False)

    #get activities (matrix of proportions)
    print("Creating Activity Matrix")

    mallet_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)

    # bayes_activity_matrix = get_activities(ldamodel)
    # gibbs_activity_matrix = get_activities(ldamallet)
    # hdp_activity_matrix = get_activities(hdpmodel)
    print("Done creating Activity Matrix")

    # bayes_activity_matrix.to_csv(output_path + project + "_LDABayes_Activities.tsv", sep="\t", index=False)
    # gibbs_activity_matrix.to_csv(output_path + project + "_LDAGibbs_Activities.tsv", sep="\t", index=False)
    # hdp_activity_matrix.to_csv(output_path + project + "_HDP_Activities.tsv", sep="\t", index=False)
