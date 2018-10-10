import re
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import operator
import csv
import sys

def wordCounter():
    """
    Get a corpus of valid words
    """
    text = (open('big.txt','r')).read()
    words = re.findall(r'\w+', text.lower())
    wordTypes_n_counts = Counter(words) # types = 32198, tokens = 1115585
    return wordTypes_n_counts

def nameList():
    """
    Get a corpus of common names
    """
    name = (open('names.txt','r').read())
    names = re.findall(r'\w+', name)
    name_counts = Counter(names)
    return name_counts

def bigramCorpus(prevWord, nextWord):
    """
    Get a corpus of precomputed bigrams

    """
    f = open('count_2w.txt','r')
    lines = f.readlines()
    bigram_dict = {}
    for line in lines:
        words = line.split()
        bigram_dict[(words[0].lower(),words[1].lower())] = int(words[2])

    if (prevWord, nextWord) not in bigram_dict.keys():
        temp = 1
        return(temp)
    else:
        return(bigram_dict[(prevWord, nextWord)])

def findCost(a, b, operation):
    """
    Return cost for insert, delete, substitute operations
    from Peter Norvig's list of errors.
    """
    insert = np.genfromtxt('insert.csv', delimiter=',')
    delete = np.genfromtxt('del.csv', delimiter=',')
    substitute = np.genfromtxt('substitute.csv', delimiter=',')
    reversal = np.genfromtxt('reversal.csv', delimiter=',')
    cost = 1
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    letter_dict = {}
    for i in range(len(letters)):
        letter_dict[letters[i]] = i
    if operation == 'insert':
        cost = int(insert[letter_dict[a], letter_dict[b]])
    if operation == 'delete':
        cost = int(delete[letter_dict[a], letter_dict[b]])
    if operation == 'substitute':
        cost = int(substitute[letter_dict[a], letter_dict[b]])
    if operation == 'reversal':
        cost = int(reversal[letter_dict[a], letter_dict[b]])
    return cost



def editDistance(word0, word1):
    """
    Find Levenshtein Distance between two words.
    Refer to Peter Norvig's cost matrix for substitution, insertion, deletion.
    """
    width = len(word0) + 1
    height = len(word1) + 1
    distance_matrix = np.zeros((width, height))

    # initialize with insertion cost
    for i in range(height):
        distance_matrix[0][i] = i
    for i in range(width):
        distance_matrix[i][0] = i

    # generate distance matrix
    ### USE COST MATRIX
    count = 0
    for i in range(1, width):
        for j in range(1, height):
            insert = distance_matrix[i, j-1] + findCost(word0[i-1],word1[j-1],'insert')
            delete = distance_matrix[i-1, j] + findCost(word0[i-1],word1[j-1],'delete')
            mismatch = distance_matrix[i-1, j-1] + 0 # if same characters, mismatch = 0
            if word0[i-1] != word1[j-1]:
                mismatch = (distance_matrix[i-1, j-1] + 
                                            findCost(word0[i-1],word1[j-1],'substitute'))
            distance_matrix[i][j] = min(insert, delete, mismatch)
        count += 1
    edit_distance = int(distance_matrix[i,j])
    return(edit_distance)


def edits1(word):
    """
    THIS FUNCTION IS BORROWED FROM PETER NORVIG.
    Generates all possible variations of a word by inserting, deleting, 
    substuting characters.
    """
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def nonWordError(word):
    """
    function that used editDistance and generates distance list.
    chooses the one with minimum edit distance
    """
    inidial_candidates = edits1(word.lower())
    nonWords_pruned = []
    validWords = wordCounter().keys()
    for i in inidial_candidates:
        if i in validWords:
            nonWords_pruned.append(i)
    if len(nonWords_pruned) == 0:
        corrected_word = word
    else:
        distances = {}
        for i in nonWords_pruned:
            distances[i] = editDistance(word, i)
            corrected_word = sorted(distances.items(), key=operator.itemgetter(1))[0][0]
    return(corrected_word)

def contextualizedCorrection(sent, words):
    """
    Real-world error correction.
    compares the sentence with few hundred thousand bigrams compiled by 
    Peter Norvig, and changes the word only if the difference between the original 
    and generated bigram counts is more than 100000.
    If there are more than one candidates, it picks a bigram randomly.
    """
    count = 0
    corrected_sentence = sent
    words = words
    while (count <= len(corrected_sentence)-2):
        prevWord = corrected_sentence[count]
        nextWord = corrected_sentence[count+1]
        validWords = wordCounter().keys()
        inidial_candidates = edits1(prevWord.lower())
        nonWords_pruned = []
        wordList = ['a','the', 'to', 'of', 'in', 'is', 'you', 'that', 'he', 'are', 'i']
        for i in inidial_candidates:
            if i in validWords:
                nonWords_pruned.append(i)
        if len(nonWords_pruned) == 0:
            prevWord = prevWord
        else:
            distances = {}
            for i in nonWords_pruned:
                if i.isalpha() and prevWord.isalpha() and len(i)>1 and len(prevWord) > 1:
                    distances[i] = editDistance(prevWord.lower(), i.lower())
                    corrected_list = distances.items()
            for i in corrected_list:
                if bigramCorpus(prevWord, nextWord) < 10:   
                    if (bigramCorpus(i[0], nextWord) - bigramCorpus(prevWord, nextWord) > 100000):
                        if (prevWord.isdigit() == False and count != 0 and 
                                                    i[0] != nextWord and i[0] not in wordList):
                            prevWord = i[0]
                            corrected_sentence[count] = prevWord
                            break
        count +=1
    return(corrected_sentence)


def nonWordErrorCorrection(sent):
    """
    Does non-word error correction. After that, it passes data to the function
    contextualCorrection, which does real-world error correction.
    """
    words = word_tokenize(sent)
    temp = ""
    valid_words = wordCounter().keys()
    names = nameList().keys()
    corrected_sentence = []
    for i,j in enumerate(words):
        if j.isalnum() == False:
            temp = j
        elif len(re.findall(r'\d+', j))>0:
            temp = j
        elif i == 0 and len(re.findall(r'\d+', j))==0:
            if j in names:
                temp = j
            elif j.lower() in valid_words:
                temp = j
            else:
                ### take care of the first uppercase letter here
                temp = nonWordError(j.lower())
                temp = temp.capitalize()
        else:
            temp = nonWordError(j.lower())
            if j in names:
                temp = j
            elif j.isupper():
                temp = j
            elif j.lower() in valid_words:
                temp = j
            else:
                if temp == j:
                    temp = j
        corrected_sentence.append(temp)
    
    # passing the sentences for centextual correction 
    corrected_sentence = contextualizedCorrection(corrected_sentence, words)

    # formatting text to show the corrected word in brackets 
    full_text = ""
    for i,j in zip(words, corrected_sentence):
        if i == j:
            full_text += i
        else:
            full_text += i + " (" + j + ") "
        full_text += " "
    return(full_text)

def main(fileLocation):
    """
    Calls the non-word correction function, which in turn calls contextual correction function.
    """
    sentences = sent_tokenize(open(fileLocation).read())
    output = ""
    for i in sentences:
        temp = nonWordErrorCorrection(i)
        output += temp
    printFile = open('output.txt','w')
    printFile.write(output)

if __name__ == "__main__":
    fileLocation = sys.argv[1]
    main(fileLocation)
