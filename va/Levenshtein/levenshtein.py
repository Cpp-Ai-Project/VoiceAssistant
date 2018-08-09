'''
Uses the Leveneshtein Algorithm(Minimum String Distance) to calculate the differnce
between a "given" string and a known string. Additionally three types of errors to help
with the testing prossess of the voice assistant. Documentation is found on our Jira site
and also for more info:

    https://www.youtube.com/watch?v=We3YDTzNXEk&t
    http://www.yorku.ca/mack/CHI01a.PDF
    https://en.m.wikipedia.org/wiki/Precision_and_recall

Used With California Polytechnic University California, Pomona Voice Assistant Project
Author: Christopher Leal
Project Manager: Gerry Fernando Patia
Writing Date: 24 May, 2018
Finished: ~7/9/2018
'''

#imports
import numpy
import sys
from colorsys import *

#constants/globals
# ___________________

def levenshtein(givenString, knownString):
    '''
    let me know what you will pass me please so I can handle it here

    the "main" function of the page
		-utilizes a matrix to calculate the levenshtein distance between the two corpus
		-prints various statements and calls error calculating methods

    :param givenString: (array or string)
    :param knownString:
    :return: int value of levenshtein value
        ---will be removed in future updates
    '''

    matrix = __generate_matrix(len(givenString) + 1, len(knownString) + 1)

    tpCheck = 0
    tp = 0
    alterations_array = __generate_alterations_array(givenString, knownString)
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

    #print_key()
    # THIS IS \/ WHERE "BUGS" WILL BE ... WILL TEST AND IMPROVE LATER
    __find_path(matrix, alterations_array)
    #__________

    #print("Known: " + knownString)
    print("Given: ", end='')
    #bugs may also \/
    __print_given_with_errors(givenString, knownString, alterations_array)
    #print(matrix)
    __print_known(knownString)


    __soukoreff_error(matrix[len(givenString)][len(knownString)],
                      max(len(givenString), len(knownString)))
    # (tp,fn,fp) \/
    __patia_formulas(tp, len(knownString) - tp, len(givenString) - tp)
    print(str(matrix[len(givenString)][len(knownString)]) + " )")

    if(type(givenString) == list):
        j=0
        for i in range(0, len(alterations_array)):
            #redundant if statement
            if(alterations_array[i] != 4):
                if(alterations_array[i] == 3):
                    __levenshtein_single(givenString[j], knownString[j])
                j+=1


    return matrix[len(givenString)][len(knownString)]


def __levenshtein_single(givenString, knownString):
    levenshtein(givenString,knownString)

# _____Error Methods_____

# i need this to be private
def __soukoreff_error(msd, maxCorpusLength) :
    '''
    :param msd: (pbv, minimum string distance, given from levenshtein)
    :param maxCorpusLength:
    :return:

    '''

    error = ( msd / maxCorpusLength ) * 100
    #print("**Soukoreff Error: " + str(error) + "%")
    print("( " + str(error) + "%" + ", ", end='')

# i need this to be private
def __patia_formulas(tp, fn, fp) :
    '''
    :param tp: (true positive)
    :param fn: (false negative)
    :param fp: (false positive)
    :return:
    '''
    
    recall = ( tp / (tp + fn) ) * 100
    precision = ( tp / (tp + fp) ) * 100
    '''
    accuracy = ( tp + tn ) / ( tp + tn + fn + fp )
    '''

    #print("**Recall: " + str(recall) + "%")
    #print("**Precision: " + str(precision) + "%")
    print(str(recall) + "%" + ", ", end='')
    print(str(precision) + "%" + ", ", end='')
    '''
    print("**Accuracy: " + str(accuracy) + "%\n")
    '''

#______Class Utilities_____
def __generate_matrix(givenLength, knownLength):
    matrix = numpy.zeros((givenLength , knownLength ), dtype=int)
    for i in range(len(matrix)):
        matrix[i][0] = i

    for i in range(len(matrix[0])):
        matrix[0][i] = i
    return matrix

def __generate_alterations_array(givenString, knownString):
    array = numpy.zeros(max(len(givenString),len(knownString)) + 5)
    for i in range(0,5):
        array[i] = 4
    return array

def print_key():
    '''
    prints the key for the levenshtein algorithm
    :return:
    '''
    print("___KEY___")
    print('\x1b[1;0;0m' + "Given char is correct" + '\x1b[0m')  # bold white
    print('\x1b[1;32;0m' + "Given char needed to be inserted" + '\x1b[0m')  # bold green
    print('\x1b[1;31;0m' + "Given char needed to be deleted" + '\x1b[0m')  # bold red
    print('\x1b[1;34;0m' + "Given char needed to be replaced" + '\x1b[0m')  # bold blue
    print("(Error%, Recall, Precision, Corpus Leven. Dist.(Overall Leven.Dist.)) ")
    print("_________")

def __print_matrix(matrix):
    for i in range(0, len(matrix)):
        for j in range (0, len(matrix[0])):
            print([i][j])

#__________________________________________________________________________

def __print_given_with_errors(givenString, knownString, alterations_array):
    '''
    prints a colored representation of string differences

    :param givenString: (pbv, can be either be an array or string)
    :param knownString: (pbv, can either be an array or string)
    :param alterations_array: (pbv, found from find_path() )
    :return:
    '''
    j=0
    k=0

    for i in range (0, len(alterations_array)):
        if(alterations_array[i]==4):
            continue
        #print("Currrent: " + str(j) + "," + str(k) + "_" + str(i))

        if(alterations_array[i] == 0):
            print('\x1b[1;0;0m' + givenString[j] + '\x1b[0m', end='')  # bold white
            j+=1
            k+=1
        elif(alterations_array[i] == 1):
            print('\x1b[1;32;0m' + knownString[k] + '\x1b[0m', end='')  # bold green
            k+=1
        elif (alterations_array[i] == 2):
            print('\x1b[1;31;0m' + givenString[j] + '\x1b[0m', end='')  # bold red
            j+=1

        elif (alterations_array[i] == 3):
            print('\x1b[1;34;0m' + givenString[j] + '\x1b[0m', end='')  # bold blue
            j+=1
            k+=1
        if(type(knownString) == list):
            print(" ",end='')

    print()

def __print_known(knownString):
    if (type(knownString) != list):
        print("Known: " + knownString)
    else:
        print("Known: ", end='')
        for i in range(0, len(knownString)):
            print(knownString[i] + " ", end='')
        print()
        
        
def __find_path(matrix, alterations_array):
    '''
    :param matrix: (pbv, leven matrix used in calculations)
    :param alterations_array: (pbr, helps with printing the color changes for 'UI')
    :return:

    finds the shortest path of alterations using the matrix made in levenshtein(2arg)
        to change the given_corpus to known_corpus
    '''
    print(matrix)
    i = len(matrix) -1
    j = len(matrix[0]) -1
    arraySlot = len(alterations_array) - 1
    while( arraySlot >= 0 ):
        #print(str(i) + " " + str(j) + " " + str(arraySlot))
        if(i ==0 and j ==0):
            break
        current = matrix[i][j]
        #print("Current Slot: " + str(i) +","+str(j)+"_"+ str(current), end=' ')

        if(current != 0):
            current -= 1

            if ( j > 0 and i > 0 and current != min(matrix[i - 1][j - 1], matrix[i][j - 1], matrix[i - 1][j]) ):
                # if this is the case we know that the letters represented by this particular slot are equiv.
                alterations_array[arraySlot] = 0
                i -= 1
                j -= 1
            else:

                if(j == 0 or current == matrix[i-1][j]):
                #elif (current == matrix[i][j - 1]):
                    alterations_array[arraySlot] = 2
                    i-=1
                elif(i == 0 or current == matrix[i][j-1]):
                #elif(current == matrix[i-1][j]):
                    alterations_array[arraySlot] = 1
                    j-=1
                elif (current == matrix[i - 1][j - 1]):
                    alterations_array[arraySlot] = 3
                    i -= 1
                    j -= 1

                else:
                    print("ERROR")

        else:
            alterations_array[arraySlot] = 0
            i -= 1
            j -= 1
        #print(str(alterations_array[arraySlot]))
        arraySlot -= 1
    print(alterations_array)
    '''
    for i in range (len(alterations_array)):
        print(alterations_array[i])
    '''
    print()

# _____Main / Testing Zone_____
def test_it():
    '''
    print("Given/Predicted String: ")
    given_corpus = sys.stdin.readline()
    print("Known String: ")
    known_corpus = sys.stdin.readline()
    '''

    #given_corpus
    #known_corpus
    given_corpus = "hi sara my are name gerry"

    known_corpus = "hello siri my name is gerry fernando patia"


    # get arrays of strings, trim last word to remove escape char
    given_corpus_array = given_corpus.split(" ")
    # given_corpus_array[len(given_corpus_array) - 1] = given_corpus_array[len(given_corpus_array) - 1][:-1]
    known_corpus_array = known_corpus.split(" ")
    # known_corpus_array[len(known_corpus_array) - 1] = known_corpus_array[len(known_corpus_array) - 1][:-1]

    #print arrays
    print("Given: ", end='')
    print(given_corpus_array)
    print("Known: ", end='')
    print(known_corpus_array)

    print_key()
    max_array_length = max(len(given_corpus_array), len(known_corpus_array))
    print("\n___Levenshtein_Claculations___")
    int = levenshtein(given_corpus_array, known_corpus_array)



#___D E M O___
test_it()


'''
Testing:
do more sub cases of words that are not same length

Known Bugs: 
an ' * ' means checked and fixed
add more if you run into any weird ones that dont make since
* "ralph", "alphabet" ...my problem child sigh...more moves than alt array provides
* "alphabet","ralph"  ...""
* "the","monster"
* "paw",awesome
* "awesome","ome" recall% ? it is kinda right more fp the possible tp

_________________________________________________________________________________________________

___Solved___
    *narrowed the number of situations where bug occurs...sorta* 
    inserted letters repeat first letter in most cases (example ralph, alphabet) 
    when j doesnt move.... line 130 the "bugs" happen it works 
    i just need another "tracker" for the known string other than j

___Solved___ the math is perfect its the coloring alg that is being sus

___Solved___ output coloration error?

___Solved___ known and given message display

___Solved___ single words fix
_________________________________________________________________________________________________

'''