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
import sys
from colorsys import *

#constants/globals


# ___________________

def levenshtein(givenString, knownString):
    matrix = generate_matrix(len(givenString)+1, len(knownString)+1)

    tpCheck = 0
    tp = 0
    alterations_array = generate_alterations_array(givenString,knownString)
    for i in range(1, len(givenString)+1):
        for j in range(1, len(knownString)+1):
            if givenString[i-1] == knownString[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
                tpCheck += 1
            else:

                minimum = min(matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1])
                matrix[i][j] = minimum + 1

        if tpCheck > 0:
            tp += 1
            tpCheck = 0

    print_key()
    # THIS IS \/ WHERE "BUGS" WILL BE ... WILL TEST AND IMPROVE LATER
    find_path(matrix, alterations_array)
    #__________

    print("Known: " + knownString)
    print("Given: ", end='')
    #bugs may also \/
    print_given_with_errors(givenString,knownString,alterations_array)
    print(matrix)
    soukoreff_error(matrix[len(givenString)][len(knownString)],
                    max(len(givenString), len(knownString)))
    # (tp,fn,fp) \/
    patia_formulas(tp,len(knownString)-tp,len(givenString)-tp)
    return matrix[len(givenString)][len(knownString)]


# _____Error Methods_____

# i need this to be private
def soukoreff_error(msd, maxCorpusLength) :

    error = ( msd / maxCorpusLength ) * 100
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

def generate_alterations_array(givenString, knownString):
    if(len(givenString) == len(knownString)):
        array = numpy.zeros(len(knownString))
    else:
        array = numpy.zeros(max(len(givenString),len(knownString)))

    return array


def print_key():
    print("___KEY___")
    print('\x1b[1;0;0m' + "Given char is correct" + '\x1b[0m')  # bold white
    print('\x1b[1;32;0m' + "Given char needed to be inserted" + '\x1b[0m')  # bold green
    print('\x1b[1;31;0m' + "Given char needed to be deleted" + '\x1b[0m')  # bold red
    print('\x1b[1;34;0m' + "Given char needed to be replaced" + '\x1b[0m')  # bold blue
    print("_________")


def print_matrix(matrix):
    for i in range(0, len(matrix)):
        for j in range (0, len(matrix[0])):
            print([i][j])

def print_given_with_errors(givenString,knownString, alterations_array):
    j=0
    for i in range (0, len(alterations_array)):
        if(alterations_array[i] == 0):
            print('\x1b[1;0;0m' + givenString[j] + '\x1b[0m', end='') # bold white
            j+=1
        elif(alterations_array[i] == 1):
            print('\x1b[1;32;0m' + knownString[j] + '\x1b[0m', end='')  # bold green
        elif (alterations_array[i] == 2):
            print('\x1b[1;31;0m' + givenString[j] + '\x1b[0m', end='')  # bold red
            j+=1
        elif (alterations_array[i] == 3):
            print('\x1b[1;34;0m' + givenString[j] + '\x1b[0m', end='')  # bold blue
            j+=1



    print()

def find_path(matrix,alterations_array):
    i = len(matrix) -1
    j = len(matrix[0]) -1
    arraySlot = len(alterations_array) - 1
    while( arraySlot >= 0 ):
        #print(str(i) + " " + str(j) + " " + str(arraySlot))
        current = matrix[i][j]
        if(current != 0):
            current -= 1
            if (current != min(matrix[i - 1][j - 1], matrix[i][j - 1], matrix[i - 1][j])):
                # if this is the case we know that the letters represented by this particular slot are equiv.

                alterations_array[arraySlot] = 0
                i -= 1
                j -= 1
            else:
                # which does it match?
                if(current == matrix[i-1][j-1]):
                    alterations_array[arraySlot] = 3
                    i -= 1
                    j -= 1

                elif(current == matrix[i][j-1]):
                    alterations_array[arraySlot] = 1
                    j -= 1

                elif(current == matrix[i-1][j]):
                    alterations_array[arraySlot] = 2
                    i -= 1


                else:
                    print("ERROR")


        else:
            alterations_array[arraySlot] = 0
            i -= 1
            j -= 1





        arraySlot -= 1

    for i in range (len(alterations_array)):
        print(alterations_array[i])


    print()

# _____Main / Testing Zone_____

print('\x1b[1;32;0m' + "Levenstein Distance is: " + str(levenshtein("apples", "oranges")) + '\x1b[0m')