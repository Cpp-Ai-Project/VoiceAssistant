'''
This program is a word embedding library, which uses natural language
processing in order to reduce the amount of training data needed to input
into the model in the form of tokens (or key words). It converts words
to vectors in order to numerically represent the learned vocabulary.
A simple user interface is utilized for ease of use such that functions
are easily accessible and the vocab is visualized in a matlab-like plot.
Performance is analyzed via a standard accuracy test from a text file
called 'question-words.txt' where the model loaded or saved is a file
named MyVectors. Consider many sources such as Project Gutenberg to
train models (free e-books) using nltk.download("gutenberg") ~ 40GB
and other corpora available from nltk or elsewhere.
'''

#for Windows users' convenience to ignore pop-up warning for "chunkize" to "chunkize_serial"
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim.models.word2vec as w2v	#Word2Vec
from gensim.models import KeyedVectors	#save,load word vector models

import codecs							#word encoding (utf-8)
import glob								#get files in directory
import nltk								#natural language toolkit (used for tokenizing sentences)
import numpy as np						#needed for np arrays
import re 								#regex
import multiprocessing					#multithreading
import os								#gives access to system events
from sklearn.manifold import TSNE		#reduce dimensions (better visual representation of vectors)
import pandas as pd						#plot labels
import texttable						#align console data
import matplotlib.pyplot as plt			#plotting

#choose to train
train_yn = ""
yn_list = ['y','Y','n','N']
while(True):
	if(train_yn in yn_list):
		break
	else:
		train_yn = input("Do you wish to train the model?(Y/N)\n")

#clean file from unicode and other unnecessary punctuation
def clean_file():
	file = input("Enter file name:\n")
	fname = file + ".txt"
	cleaned = file + "Cleaned.txt"

	f = open(fname,'r')
	statement = f.read()
	#print(statement)
	pos = 0
	output = ""

	for c in statement:
		c = ord(c)
		pos += 1
		#if char not alphabetical, space, new line, or period, make it null
		if (c < 97 or c > 122) and (c < 65 or c > 90) and c != 32 and c != 46 and c != 10:
			c = 0;
		#make letters lowercase
		if(c >= 65 and c <= 90):
			c += 32
		if c != 0:
			c = chr(c)
			output += c

	print(output + "\n")
	f = open(cleaned,'w')
	f.write(output)
	print("File cleaned of unnecessary characters and\nsaved as [name]Cleaned.txt")

def read_tokenize():
	#read text file with asserted name ("*.txt" if all are in directory are read)
	file = input("Enter file name: (all for all files in directory)\n")
	fname = file + ".txt"

	if(file == "all"):
		book_name = glob.glob("*.txt")
		corpus_raw = u"" #u = unicode string
	else:
		book_name = glob.glob(fname)
		corpus_raw = u"" #u = unicode string
	for book_filename in book_name:
		print("Reading '{0}'...".format(book_filename))
		with codecs.open(book_filename, "r", "utf-8") as book_file: #convert book file into utf format using codecs
			corpus_raw += book_file.read()
		print("Corpus is now {0} characters long".format(len(corpus_raw)))
		print()

	#convert text into sentences, tokenize
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') #import trained tokenizer's through pickle file
	raw_sentences = tokenizer.tokenize(corpus_raw)
	return raw_sentences,corpus_raw

def sentence_to_wordlist(raw):
	clean = re.sub("[^a-zA-Z]"," ", raw) #regular expression to split into words
	words = clean.split()
	return words

def count_tokens():
	#sentence where each word is tokenized
	sentences = []
	for raw_sentence in raw_sentences: 
		if len(raw_sentence) > 0: 
			sentences.append(sentence_to_wordlist(raw_sentence))

	#print number of unique tokens
	token_count = sum([len(sentence) for sentence in sentences])
	print("The book corpus contains {0:,} tokens".format(token_count))
	return sentences

def init_and_train():
	# Dimensionality of the resulting word vectors
	num_features = 100
	# Minimum word count threshold.
	min_word_count = 3
	# Number of threads to run in parallel.
	num_workers = multiprocessing.cpu_count()
	# Context window length.
	context_size = 7
	# Downsample setting for frequent words; how often you look at the same word
	# the more frequent a word is, the less you use it to create a vector
	#0 - 1e-5, where "0" is off by default
	downsampling = 1e-3
	# Seed for the RNG, to make the results reproducible.
	#deterministic, good for debugging
	seed = 1

	#model from gensim w2v model
	corpus_count = len(corpus_raw)

	word2vector = w2v.Word2Vec(
		sg=1,
		seed=seed,
		workers=num_workers,
		size=num_features,
		min_count=min_word_count,
		window=context_size,
		sample=downsampling,
		sorted_vocab=1
	)
	return corpus_count, num_features, word2vector

def build():	
	#build vocab
	sentences = count_tokens()
	word2vector.build_vocab(sentences)
	if(train_yn == "y" or train_yn == "Y"):
		word2vector.train(sentences, total_examples=len(corpus_raw), epochs=1000)

	print("Word2Vec vocabulary length:", len(word2vector.wv.vocab))

def update_vocab(corpus_count):
	sentences = count_tokens()
	word2vector.build_vocab(sentences, update = True)
	if(train_yn == "y" or train_yn == "Y"):
		word2vector.train(sentences, total_examples=corpus_count, epochs=1000)
	print("Word2Vec vocabulary length:",  len(word2vector.wv.vocab), "\n")
	return corpus_count

def save_load(word2vector):
	if not os.path.exists("TrainedModels"):
		os.makedirs("TrainedModels")
	if(train_yn == "y" or train_yn == "Y"):
		word2vector.save("MyVectors")
	else:
		word2vector = KeyedVectors.load("MyVectors")
	return word2vector

def plot():
	#gather vector data
	word_vectors_matrix = []
	n_points = int(input("Enter number of word vectors do you want to see: "))
	plot_count = 0
	for item in word2vector.wv.vocab:
		word_vectors_matrix.append(word2vector[item])
		plot_count += 1
		if(plot_count == n_points or plot_count == len(word2vector.wv.vocab)):
			break
	#reduce to lower dimension (2d) via t-SNE
	tsne = TSNE(n_components=2)
	word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix) #transform tsne model

	vocab = list(word2vector.wv.vocab)
	df = pd.DataFrame(word_vectors_matrix_2d, index=vocab[0:n_points], columns=['x', 'y'])
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	
	ax.scatter(df['x'], df['y'])
	for word, pos in df.iterrows():
		ax.annotate(word, pos)
	plt.show()

def plot_n(pos_list,neg_list,result):
	'''myindex = pos_list + neg_list
	myindex.append(result)'''
	res_list = []
	res_list.append(result)

	plot_arr = []
	plot_pos = []
	plot_neg = []
	plot_res = []

	for item in word2vector.wv.vocab:
		if(item in pos_list or item in neg_list or item == result):
			plot_arr.append(word2vector[item])

	#reduce to lower dimension (2d) via t-SNE
	tsne = TSNE(n_components=2)
	plot_arr_2d = tsne.fit_transform(plot_arr) #transform tsne model
	
	#separate data into (+),(-),(=) again
	plot_pos = plot_arr_2d[:len(pos_list)]
	plot_neg = plot_arr_2d[len(pos_list):len(pos_list) + len(neg_list)]
	plot_res = plot_arr_2d[len(pos_list) + len(neg_list):]

	df_p = pd.DataFrame(plot_pos, index=pos_list, columns=['x', 'y'])
	df_n = pd.DataFrame(plot_neg, index=neg_list, columns=['x', 'y'])
	df_r = pd.DataFrame(plot_res, index=res_list, columns=['x', 'y'])

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	ax.scatter(df_p['x'], df_p['y'], c='green')
	for word, pos in df_p.iterrows():
		ax.annotate(word, pos)
	ax.scatter(df_n['x'], df_n['y'], c='red')
	for word, pos in df_n.iterrows():
		ax.annotate(word, pos)
	ax.scatter(df_r['x'], df_r['y'], c='black')
	for word, pos in df_r.iterrows():
		ax.annotate(word, pos)
	plt.show()

def nearest_similarity_cosmul(start1, end1, end2):
	#alternative way to mathematically find suggested word similar to "start1 + end2 - end1"
	similarities = word2vector.wv.most_similar_cosmul(
		positive=[end2, start1],
		negative=[end1]
	)
	start2 = similarities[0][0]
	print("{start1} is related to '{end1}', as '{start2}' is related to '{end2}'".format(**locals()))
	return start2



#begin, initialize model
raw_sentences, corpus_raw = read_tokenize()
corpus_count, dimensions, word2vector = init_and_train()
build()
word2vector = save_load(word2vector)
#print out interface for user decision
print("\nWhat do you wish to do? (upper or lowercase)")
table = texttable.Texttable()
table.set_cols_align(["c","c"])
table.add_rows(
	[["Key","Function"],
	["2","compare 2 words"],
	["A","word analogy"],
	["C","clean file"],
	["E","exit program"],
	["N","n closest words"],
	["M","word math"],
	["PERF","performance eval"],
	["PLOT","default vector plot"],
	["T","train further"],
	["w2v","word -> vec"],
	["v2w","vec -> word"]])	

#runner
while(True):
	print()
	print(table.draw())
	decision = input(">> ")
	if(decision == "c" or decision == "C"):
		clean_file()

	elif(decision == "plot" or decision == "PLOT"):
		plot()

	elif(decision == "plotn" or decision == "PLOTN"):
		po = ['king','woman']
		ne = ['man']
		re = 'queen'
		plot_n(po,ne,re)

	elif(decision == "perf" or decision == "PERF"):
		'''note: all words need to be in vocab,
				 consider training model on
				 questions-words.txt first '''
		model.accuracy('/tmp/questions-words.txt')

	elif(decision == "t" or decision == "T"):
		save_count = corpus_count
		raw_sentences, corpus_raw = read_tokenize()
		save_count = corpus_count + len(corpus_raw)
		corpus_count = save_count
		#choose to train
		train_yn = ""
		yn_list = ['y','Y','n','N']
		while(True):
			if(train_yn in yn_list):
				break
			else:
				train_yn = input("Do you wish to train the model?(Y/N)\n")
		update_vocab(corpus_count)
		save_load(word2vector)
		
	elif(decision == "w2v"):
		word = input("Which word do you want to convert?\n")
		if word in word2vector.wv.vocab:
			print(str(word2vector[word]) + "\n")
		else:
			print(word, "is not in the model's vocabulary.")
	
	elif(decision == "v2w"):
		element = ""
		my_vector = []
		my_dimensions = dimensions
		print("Enter a " + str(dimensions) + "-D array:")
		while(my_dimensions != 0):
			element = int(input("Element " + str(11-my_dimensions) + ": "))
			my_dimensions -= 1
			my_vector.append(element)
		model_word_vector = np.array(my_vector,dtype='f')
		print("The word representation for", str(my_vector), "is '", 
			word2vector.most_similar(positive=[model_word_vector], topn=1)[0][0],
			"' with a similarity of",
			round(100*word2vector.most_similar(positive=[model_word_vector], topn=1)[0][1],2), "%")
		
	elif(decision == "n" or decision == "N"):
		n = int(input("Enter n: "))
		nword = input("Type a word to find the " + str(n) + " closest words to it: ")
		print("The " + str(n) + " closest word(s) to '" + nword + "' is(are):")
		for i in range(n):
			print(word2vector.most_similar(positive=[nword], topn=n)[i][0],end=", ")
		print("with similarities of:")
		for j in range(n):
			print(round(100*word2vector.most_similar(positive=[nword], topn=n)[j][1],2),"%",end=", ")
		print()
		
	elif(decision == "2"):
		word1 = input("Enter word 1: ")
		word2 = input("Enter word 2: ")
		print("Similarity between '" + word1 + "' and '" + word2 + "' is: " + 
			str(word2vector.wv.similarity(word1, word2)) + "\n")

	elif(decision == "a" or decision == "A"):
		#note: very inaccurate with smaller models
		print("Enter words A,B,C in format as follows:\n" +
			"A is to B as C is to _")
		A = input("A:")
		B = input("B:")
		C = input("C:")
		if(A in word2vector.wv.vocab):
			print(A, "is in the vocab.")
		else:
			print(A, "is not in the vocab.")
			break
		if(B in word2vector.wv.vocab):
			print(B, "is in the vocab.")
		else:
			print(B, "is not in the vocab.")
			break
		if(C in word2vector.wv.vocab):
			print(C, "is in the vocab.")
		else:
			print(C, "is not in the vocab.")
			break

		Av = word2vector[A]
		Bv = word2vector[B]
		Cv = word2vector[C]
		
		ABdist = Bv-Av
		Dv = Cv + ABdist

		D = word2vector.most_similar(positive=[Dv], topn=1)[0][0]

		print(A, "is to", B, "as", C, "is to", D)

	elif(decision == "m" or decision == "M"):
		positive_list = []
		negative_list = []
		pos = ""
		neg = ""
		p_index = 0
		n_index = 0
		while(pos != "q" and pos != "Q"):
			pos = input("Enter positive word(+) or 'Q' to quit: ")
			if(pos != "q" and pos != "Q"):
				positive_list.append(pos)
		while(neg != "q" and neg != "Q"):
			neg = input("Enter negative word(-) or 'Q' to quit: ")
			if(neg != "q" and neg != "Q"):
				negative_list.append(neg)
		print()
		for p in positive_list:
			p_index += 1
			if(p_index == len(positive_list)):
				print(p, end="")
			else:
				print(p + " + ",end="")
		print(" - ", end="")
		for n in negative_list:
			n_index += 1
			if(n_index == len(negative_list)):
				print(n, end="")
			else:
				print(n + " - ",end="")
		ans = word2vector.wv.most_similar(positive=positive_list, negative=negative_list,topn=1)

		print(" =", ans[0][0], "with a similarity of", round(100*ans[0][1],2),"%")
		
		#plot
		plot_yn = ""
		while(True):
			if(plot_yn in yn_list):
				break
			else:
				plot_yn = input("Do you wish to see the graph?(Y/N)\n")

		if plot_yn == "y" or plot_yn == "Y":
			plot_n(positive_list, negative_list, ans[0][0])

	elif(decision == "e" or decision == "E"):
		break

	else:
		continue''