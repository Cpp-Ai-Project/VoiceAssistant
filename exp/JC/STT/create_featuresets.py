#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import glob

from utils import sparse_tuple_from as sparse_tuple_from

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

def getTrainingData():
	#print("Training Data")
	train_x_input = []
	train_x_seq = []
	train_y = []
	os.chdir("C:\\Users\\Jason Chang\\Documents\\Cal Poly Pomona\\CPP AI\\STT\\LibriSpeech\\dev-clean-wav\\")

	#Finds all the audio files and put it into an array
	audio_filenames = []
	text_filenames = []
	for file in glob.glob("*.wav"):
		audio_filenames.append(file)
	for file in glob.glob("*.txt"):
		text_filenames.append(file)

	#print(audio_filenames)
	#print(text_filenames)
	
	#Gets the audio files and put them in another train_x array
	for audio_filename in audio_filenames:
		# Load wav files
		fs, audio = wav.read(audio_filename)
		
		# Get mfcc coefficients
		inputs = mfcc(audio, samplerate=fs)

		# Tranform in 3D array
		train_inputs = np.asarray(inputs[np.newaxis, :])
		train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
		train_seq_len = [train_inputs.shape[1]]
		train_x_input.append(train_inputs)
		train_x_seq.append(train_seq_len)
	for text_filename in text_filenames:
		# Readings text file
		with open(text_filename, 'r') as f:

			#Only the last line is necessary
			line = f.readlines()[-1]

			# Get only the words between [a-z] and replace period for none
			original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
			targets = original.replace(' ', '  ')
			targets = targets.split(' ')
			#print(targets)

			# Adding blank label
			targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

			# Transform char into index
			targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

			# Creating sparse representation to feed the placeholder
			train_targets = sparse_tuple_from([targets])
			train_y.append(train_targets)

	return train_x_input, train_x_seq, train_y			####ADDS ANOTHER ROW#####

def getTestData():
	#print("Test Data")
	test_x_input = []
	test_x_seq = []
	test_y = []
	os.chdir("C:\\Users\\Jason Chang\\Documents\\Cal Poly Pomona\\CPP AI\\STT\\LibriSpeech\\test-clean-wav\\")

	#Finds all the audio files and put it into an array
	audio_filenames = []
	text_filenames = []
	for file in glob.glob("*.wav"):
		audio_filenames.append(file)
	for file in glob.glob("*.txt"):
		text_filenames.append(file)

	#print(audio_filenames)
	#print(text_filenames)
	
	#Gets the audio files and put them in another train_x array
	for audio_filename in audio_filenames:
		# Load wav files
		fs, audio = wav.read(audio_filename)
		
		# Get mfcc coefficients
		inputs = mfcc(audio, samplerate=fs)

		# Tranform in 3D array
		test_inputs = np.asarray(inputs[np.newaxis, :])
		test_inputs = (test_inputs - np.mean(test_inputs))/np.std(test_inputs)
		test_seq_len = [test_inputs.shape[1]]
		test_x_input.append(test_inputs)
		test_x_seq.append(test_seq_len)

	for text_filename in text_filenames:
		# Readings text files
		with open(text_filename, 'r') as f:

			#Only the last line is necessary
			line = f.readlines()[-1]

			# Get only the words between [a-z] and replace period for none
			original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
			targets = original.replace(' ', '  ')
			targets = targets.split(' ')
			#print(targets)

			# Adding blank label
			targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

			# Transform char into index
			targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

			# Creating sparse representation to feed the placeholder
			test_targets = sparse_tuple_from([targets])
			test_y.append(test_targets)

	return test_x_input, test_x_seq, test_y

def create_feature_sets_and_labels():
	train_x_input, train_x_seq, train_y = getTrainingData()
	test_x_input, test_x_seq, test_y =getTestData()

	return train_x_input, train_x_seq, train_y, test_x_input, test_x_seq, test_y




###Just for one sample from dev-clean-wav folder and one from test-clean-wav
def getData():
	#getting Training data
	os.chdir("C:\\Users\\Jason Chang\\Documents\\Cal Poly Pomona\\CPP AI\\STT\\LibriSpeech\\dev-clean-wav\\")

	# Load wav files
	fs, audio = wav.read("3752-4944-0041.wav")
		
	# Get mfcc coefficients
	inputs = mfcc(audio, samplerate=fs, numcep = 26)

	# Tranform in 3D array
	train_inputs = np.asarray(inputs[np.newaxis, :])
	train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
	train_seq_len = [train_inputs.shape[1]]

	# Readings text file
	with open("3752-4944-0041.txt", 'r') as f:

		#Only the last line is necessary
		line = f.readlines()[-1]

		# Get only the words between [a-z] and replace period for none
		original_train = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
		targets = original_train.replace(' ', '  ')
		targets = targets.split(' ')
		#print(targets)

		# Adding blank label
		targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

		# Transform char into index
		targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

		# Creating sparse representation to feed the placeholder
		train_targets = sparse_tuple_from([targets])
		
	#Getting testing data
	os.chdir("C:\\Users\\Jason Chang\\Documents\\Cal Poly Pomona\\CPP AI\\STT\\LibriSpeech\\dev-clean-wav\\")

	# Load wav files
	fs, audio = wav.read("3752-4944-0041.wav")
		
	# Get mfcc coefficients
	inputs = mfcc(audio, samplerate=fs, numcep = 26)

	# Tranform in 3D array
	test_inputs = np.asarray(inputs[np.newaxis, :])
	test_inputs = (test_inputs - np.mean(test_inputs))/np.std(test_inputs)
	test_seq_len = [test_inputs.shape[1]]

	# Readings text file
	with open("3752-4944-0041.txt", 'r') as f:

		#Only the last line is necessary
		line = f.readlines()[-1]

		# Get only the words between [a-z] and replace period for none
		original_test = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
		targets = original_test.replace(' ', '  ')
		targets = targets.split(' ')
		#print(targets)

		# Adding blank label
		targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

		# Transform char into index
		targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in targets])

		# Creating sparse representation to feed the placeholder
		test_targets = sparse_tuple_from([targets])

	return train_inputs, train_seq_len, train_targets, test_inputs, test_seq_len, test_targets, original_test