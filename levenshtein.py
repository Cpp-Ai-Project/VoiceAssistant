'''
Uses the Leveneshtein Algorith(Minimum String Distance) to calculate the differnce
between a "given" string and a known string. Additionally three types of errors to help
with the testing prosses of the voice assistant. Documentation is found on our Jira site
and also for more info:

    https://www.youtube.com/watch?v=We3YDTzNXEk&t
    http://www.yorku.ca/mack/CHI01a.PDF
    https://en.m.wikipedia.org/wiki/Precision_and_recall

Used With California Polythechnic University California, Pomona Voice Assistant Project
Author: Christopher Leal
Project Manager: Gerry Fernando Patia
Date: 24 May, 2018
'''

#imports
import numpy

#constants


# ___________________

def levenshtein(givenString, knownString):
    matrix = generate_matrix(len(givenString)+1, len(knownString)+1)
    tpCheck = 0
    tp = 0
    for i in range(1, len(givenString)+1):
        for j in range(1, len(knownString)+1):
            if givenString[i-1] == knownString[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
                tpCheck += 1
            else:
                matrix[i][j] = min(matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1]) +1
        if tpCheck > 0:
            tp += 1
            tpCheck = 0

    print(matrix)
    soukoreff_error(matrix[len(givenString)][len(knownString)],
                    max(len(givenString), len(knownString)))
    # (tp,fn,fp) \/
    patia_formulas(tp,len(knownString)-tp,len(givenString)-tp,)
    return matrix[len(givenString)][len(knownString)]


# _____Error Methods_____

# i need this to be private
def soukoreff_error(msd, maxInt) :

    error = ( msd / maxInt ) * 100
    print("**Soukoreff Error: " + str(error) + "%")

# i need this to be private
def patia_formulas(tp, fn, fp) :

    recall = ( tp / (tp + fn) ) * 100
    precision = ( tp / (tp + fp) ) * 100
    '''
    accuracy = ( tp + tn ) / ( tp + tn + fn + fp )
    '''

    print("**Recall: " + str(recall) + "%")
    print("**Precision: " + str(precision) + "%")
    '''
    print("**Accuracy: " + str(accuracy) + "%\n")
    '''



# _____Class Utilities_____
def generate_matrix(givenLength,knownLength):
    matrix = numpy.zeros((givenLength , knownLength ), dtype=int)
    for i in range(len(matrix)):
        matrix[i][0] = i

    for i in range(len(matrix[0])):
        matrix[0][i] = i

    return matrix

def print_matrix(matrix):
    for i in range(0, len(matrix)):
        for j in range (0, len(matrix[0])):
            print([i][j])

# _____Main / Testing Zone_____
print( str( levenshtein("apples", "oranges") ))
