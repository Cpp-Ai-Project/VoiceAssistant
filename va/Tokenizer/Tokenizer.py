'''
Not sure about this atm.heres what i know:

    Tokenizer to transform numbers into their spoken form.
        Ex: 11 -> eleven
    At some point I may rename this and expand to abbreviations
        Ex: Mr. -> Mister


Used With California Polytechnic University California, Pomona Voice Assistant Project
Author: Christopher Leal
Project Manager: Gerry Fernando Patia
Writing Date: 21 July, 2018
Finished:
'''

# imports



# constants/globals
ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
TEENS = ['ten' , 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen'
        , 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['\b' , 'ten', 'twenty', 'thirty', 'fourty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
MODIFIERS = ['negative', 'positive', 'hundred', 'thousand', 'million', 'billion']
__word_num_array = []

# how big of numbers are we expecting?


# ________________

def tokenize_number2(numString):
    for i in range (0,len(numString)):
        if(int(numString[0]) == 0):
            numString=numString[1:]
    print(numString)

def tokenize_number(numString):
    #negative / positive not in yet
    #print(numString)
    for i in range (0,len(numString)):
        if(int(numString[0]) == 0):
            #print(numString[1:])
            numString=numString[1:]


    if(numString != ''):
        if(int(numString) < 100):
            if(len(numString) % 3 == 1):
                __word_num_array.append(ONES[int(numString[0])])
            else:
                if(int(numString[0]) == 1):
                    __word_num_array.append(TEENS[int(numString[1])])
                else:
                    __word_num_array.append(TENS[int(numString[0])])
                    if(int(numString[1])!= 0):
                        __word_num_array.append(ONES[int(numString[1])])
                        #print(__word_num_array)

        else:
            if (int(numString) < 1000):
                __word_num_array.append(ONES[int(numString[0])])
                __word_num_array.append(MODIFIERS[2])
                tokenize_number(numString[1:])
            else:
                lendiv3= len(numString) // 3
                lenmod3 = len(numString) % 3
                if(lenmod3 == 0):
                    lenmod3 = 3
                store_num = numString[lenmod3:]
                tokenize_number(numString[:lenmod3])

                if(lenmod3 != 3):
                    __word_num_array.append(MODIFIERS[lendiv3 + 2])
                else:
                    __word_num_array.append(MODIFIERS[lendiv3 + 1])
                tokenize_number(store_num)

    #print(__word_num_array)
    '''
        if(int(numString) < 1000):
            __word_num_array.append(ONES[int(numString[0])])
            __word_num_array.append(MODIFIERS[lenmod3 + 2])
            tokenize_number(numString[1: ])
    '''


def tokenize_corpus(corpus):
    i = 0
    store_word = ''
    num_word = ''
    words = []
    while(i < len(corpus)):
        #ascii values for punctuation and "symbols" falls between dec 32 and 64
        if(32 <= ord(corpus[i]) < 64 or 91<= ord(corpus[i]) < 97):
            #numbers are between 48 and 58
            if(48<= ord(corpus[i]) < 58):
                num_word += corpus[i]
            else:
                if(store_word != ''):
                    words.append(store_word)
                    store_word=''
                if(num_word != ''):
                    tokenize_number(num_word)
                    for eachword in __word_num_array:
                        words.append(eachword)
                    num_word=''
        else:
            #assume letter
            store_word += corpus[i]
        i += 1
    return words




def __concat_words(strArray):
    string=""
    for i in range (0, len(strArray)):
        string += strArray[i]
        if(i+1 < len(strArray)):
            string += " "
    global __word_num_array
    __word_num_array = []
    return string

# ____ Main / Testing Zone ____

def __test_it(sumNumStr):
    #print(str(sumNumStr) + " -> ",end='')
    tokenize_number(sumNumStr)
    print(__concat_words(__word_num_array))
    print( tokenize_corpus("Hello, my name is Chris Leal. I am a computer programmer. I am also 23 years old.") )


#   D E M O


__test_it('1000123')
'''
Notes:

___Solved___ numbers / 100 have an issue...otherwise everything is good

___Solved___ bug with 1,000,123

make it so that the tokenize_number returns the string!!!

if a corpus has multiple strings...how do we want to handle that???

___Solved___ first if with return is bad code!

___Solved___ tokenize sentence by sentence and also individual words
                periods . and apostrophies ' are "delimiters"
    ****Special Cases may arise that will need to be added

money and degrees?



'''