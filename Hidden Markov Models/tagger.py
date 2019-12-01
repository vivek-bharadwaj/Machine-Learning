import numpy as np

from util import accuracy
from hmm import HMM


def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	S = len(tags)
	L = len(train_data)
	words_dict = dict()					# Assign unique number to each word
	unique_words = list()				# List of unique words
	states_dict = dict()
	initial_tag_counts = dict()
	tag_counts_dict = dict()
	tag_words_dict = dict()
	tag_tags_dict = dict()

	pi = None
	A = None
	B = None

	initial_tag_counts = dict.fromkeys(tags, 0)
	states_dict = {
		tags[i]: i for i in range(S)
	}
	tag_counts_dict = dict.fromkeys(tags, 0)
	tag_words_dict = {
		tags[i]: {} for i in range(S)
	}
	tag_tags_dict = {
		tags[i]: {
			tags[j]: 0 for j in range(S)
		} for i in range(S)
	}

	for line in train_data:
		first_tag = line.tags[0]
		initial_tag_counts[first_tag] += 1

		for i in range(line.length):
			word, tag = line.words[i], line.tags[i]
			tag_counts_dict[tag] += 1

			if word not in words_dict:
				words_dict[word] = len(unique_words)
				unique_words.append(word)

			for k in tag_words_dict:
				if word in tag_words_dict[tag]:
					tag_words_dict[tag][word] += 1
				else:
					tag_words_dict[tag][word] = 0

			if i < line.length - 1:
				tag_tags_dict[tag][line.tags[i + 1]] += 1

	init_tags_count_list = list(initial_tag_counts[tag] for tag in tags)
	denominator = list([tag_counts_dict[tag]] for tag in tags)
	tag1_tag2_list = list([tag_tags_dict[tag1].get(tag2, 1e-6) for tag2 in tags] for tag1 in tags)
	tag_words_list = list([tag_words_dict[tag].get(word, 1e-6) for word in unique_words] for tag in tags)

	pi = np.array(init_tags_count_list) / L
	A = np.array(tag1_tag2_list) / np.array(denominator)
	B = np.array(tag_words_list) / np.array(denominator)
	model = HMM(pi, A, B, obs_dict=words_dict, state_dict=states_dict)
	###################################################
	return model


def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	rows, cols = model.B.shape[0], model.B.shape[1]
	column_to_add = np.ones((rows, 1))

	for line in test_data:
		for word in line.words:
			if word not in model.obs_dict:
				model.obs_dict[word] = len(model.obs_dict)
				new_observation = column_to_add * 1e-6
				model.B = np.hstack((model.B, new_observation))
		tagging.append(model.viterbi(obs_sequence=line.words))
	###################################################
	return tagging
