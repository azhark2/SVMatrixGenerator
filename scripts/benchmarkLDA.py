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
	for i, col in enumerate(df.columns[1:]): #for every sample
		counts = list(df[col])
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


output_path = "/Users/azhark/iCloud/dev/SVMatrixGenerator/results/LDA/Benchmarks/"

paths = ["/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_1/ground.truth.syn.catalog.csv", 
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_2/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_3/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_4/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_5/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_6/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_7/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_8/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_9/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_10/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_11/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_12/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_13/ground.truth.syn.catalog.csv",
			"/Users/azhark/iCloud/dev/SVMatrixGenerator/data/Benchmark_Kit/Data/14Scenarios/scenario_14/ground.truth.syn.catalog.csv"]

#loop through input matrices of cancer types and perform LDA
for path in paths:
	project = path.split("/")[:-1].join("/") + "ground.truth.syn.sigs.csv"
	print("Executing LDA algorithm on " + project)
	
	pd.read_csv()
	num_sigs = projectToSigs[project]
	bagOfMutations, idToChannel = matrixToBag(path)
	classification = list(pd.read_csv(path, sep="\t", usecols=[0]).iloc[:, 0])


	print("Extracting Bayes Signatures")
	ldamodel = LdaMulticore(bagOfMutations, num_topics=num_sigs, id2word = idToChannel, passes=100, iterations=100, minimum_probability=0)
	
	print("Now Extracting Gibbs Signatures")
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
	

	gibbs_signatures.index = channels
	
	gibbs_signatures = gibbs_signatures.loc[classification, :]


	bayes_signatures.reset_index(inplace=True)
	bayes_signatures.to_csv(output_path + project + "_LDABayes_Signatures.tsv", sep="\t", index=False)

	gibbs_signatures.reset_index(inplace=True)
	gibbs_signatures.to_csv(output_path + project + "_LDAGibbs_Signatures.tsv", sep="\t", index=False)
