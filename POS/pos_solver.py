#!/usr/bin/env python3
###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: 
# Satyaraja Dasara - sdasara
# Abhinav Reddy Kaitha - abkaitha106
# Justin Lee - jl146
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
############# (1) a description of how you formulated the problem, including precisely defining the abstractions ##############
#
######### CALCULATIONS DONE IN TRAIN FUNCTION:################################################################################
#
# First we counted the number of times a particular parts of speech is occuring at the first position of a sentence. This is 
# stored in the initial dictionary
#
# All the dictionaries used for calculations are not initialised with keys. So I have used try and except, since an error occurs
# when ever a new key is generated.
#
# In the same loop we calculated the total nummber of times a particular parts of speech is occuring in the whole training data
#  This is stored in pos dictionary
#
# Then we counted the number of times a particular parts of speech is occuring after a parts of speech.
# This is stored in transition dictionary
#
# Then we counted the number of word that belong to a particular parts of speech and stored in emission dictionary
#
# Then we changed all these counts to probabilities.
#
# To calculate p(Si/Si-1,Si-2) which is used in mcmc, we created a zip on list of parts of speech taking 3 successively
# We counte the occurance of a parts of speech after two parts of speech.
# Then we changed all these counts to probabilities and stored in the same dictionay triple.
#
######### SIMPLIFIED  ##########################
#
# In this for each word in the test sentence We have calcuated the p(word/parts of speech)p(parts of speech) with all the parts of speech.
#
## If a particular word in test is not present in train, we assigned a samll value(1/10**6) to p(word/parts of speech)
#
# We took the maximum value obtained for each word and returned that particular parts of speech.
# We then returned sum of log of all the maximum values for a sentence to the posterior function.
#
######### MCMC ################################
#
## We initialised the test sentence's parts of speech as ['noun']*length of sentence given as input
#
## Based on trial and error and considering time constraints, we have decided to generate 100 samples.
# 
# First we considered a parts of speech("part") for the first word. Then we calculate p(word1/part)*initial_prob(part)
# 
# For the second word when we assign a new pos "part" instead of initial probability we consider transition(pos of 2nd ="part"/pos of 1st)
#
# For the remaining words, if we assign "part" to ith word we consider the transition p(pos of i = "part" /pos of i-1, pos of i-2)
#
# We also calculate the p(pos of i+1 / pos of i, pos of i-1 ) and p(pos of i+2 / pos of i+1, pos of i ) for all word postions except last and second last
#
# For second last position we calculate p(pos of i+1 / pos of i) i.e transition probability
#
# In every iteration, after trying all the pos for a particular word we are using np.random.choice(len(parts),p = np.array(probability_values))
# this gives us a random index to assign a parts of speech, but this randomness is based on the probabilities obtained above
# We append this state to the sample list and proceede to generate further samples.
#
## In the process of above calculations, if any of the probabilities are not present, we assigned it to a samll value.
#
## Based on trial and error, we have excluded the first 50 samples. From the remaining samples, for a particular word, we took
# the pos that is occuring most number of times and returned it.
#
# Once the pos are finalised, we are calcuating the log(posterior) ain the same way as above and returning it.
#
######### viterbi algorithm #####################
# 
# We have intialized two arrays   viterbi_table = np.zeros((len(parts),len(sentence)))
#                                 trace = np.zeros((len(parts),len(sentence)),dtype=int)
# First one(viterbi table) to store the log(posterior) values of all possible pos for a particular word
# Second one is used to store the index of value of posterior in the previous column which resulted in the maximum 
# emission* transition value. This is used for back tracing.
#
## In the process of calcuating the probabilities, if any of the values are not found, we initiaze them to a samll value.
# For back tracing, We found the maximum value in the lat column of the veterbi table, then we returned the pos of last word
# as the pos[index in the last column where it is maximum]
# then we passed this index to the last column of trace and got the index in the penultimate colum which id used to obtain the 
# maximum value in the last column.
#
# We repeated this prosecc till the end and returens corresponsing pos and posterior values.
#
########### (2) Description of how the program works ######################################################################
# 
# label.py is the main program file
# The input to this program is a training file with each line consisting of a word, followed by a space, 
# followed by its parts of speech
#
# This command can be used to run this on silo ./label.py training_file testing_file
#
# The training data is change into list of nested tuples, where each tuple consists of two tuples. First one has words
# and the second one has its corresponding pos.
#
# Each sentence is passed to the solver function in the pos_solver file and the above calculations are done.
#
# Even the test file is splitted into individual sentences and passed.
#
########### (3) Disscussion of problems, assumptions, simplification, and design decisions we made ########################
# 
# #### In simplified
# If a particular word in test is not present in train, we assigned a small value(1/10**6) to p(word/parts of speech)
#
# #### In MCMC
# We initialised the test sentence's parts of speech with the result obtained from the simplified algorithm
# Based on trial and error and cosidering time constraints, we have decided to generate 200 samples
# In the process of  calculations, if any of the probabilities are not present, we assigned it to a small value(1/10**6).
# Based on trial and error, we have excluded the first 101 samples.
# 
# #### In viterbi
# In the process of calcuating the probabilities, if any of the values are not found, we initiaze them to a small value(1/10**6).
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct: 
#   0. Ground truth:      100.00%              100.00%
#         1. Simple:       93.92%               47.45%
#            2. HMM:       94.88%               53.55%
#        3. Complex:       90.18%               34.05%
####

import random
import math
from math import log
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    def __init__(self):
        self.pos = {}
        self.initial = {}
        self.emission = {}
        self.transition = {}
        self.small_value = 1/10**6
        self.simple_post = 0
        self.viterbi_post = 0
        self.mcmc_post = 0
        self.triple = {}
        self.samples = []


    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.simple_post
        elif model == "Complex":
            return self.mcmc_post
        elif model == "HMM":
            return self.viterbi_post
        else:
            return -999

    # Do the training!
    #
    def train(self, data):

        # PARTS OF SPEECH AND INITIAL COUNT

        for line in data:

            # Calculating initial probabilities
            try:
                self.initial[line[1][0]] +=1
            except:
                self.initial[line[1][0]] = 1

            for part in line[1]:

                try:
                    self.pos[part]+=1
                except:
                    self.pos[part]=1

        # TRANSITION
        self.transition = {x: {y: self.small_value for y in self.pos.keys()} for x in self.pos.keys()}

        for line in data:
            for one,two in zip(line[1][0:len(line[1])-1],line[1][1:]):

                try:
                    self.transition[one][two]+=1
                except:

                    try:
                        self.transition[one][two]=1
                    except:
                        self.transition[one] = {}

        # EMISSION


        for parts_speech in self.pos.keys():
            self.emission[parts_speech] = {}

        for line in data:
            for word, part in zip(line[0],line[1]):

                try:
                    self.emission[part][word] += 1
                except:
                    self.emission[part][word] = 1


         # PROBABILITIES

        # initial

        total_initial = sum(self.initial.values())

        for i in self.initial:
            self.initial[i] /= total_initial

        # parts of speech

        total_pos = sum(self.pos.values())

        for i in self.pos:
            self.pos[i] /= total_pos

        # transition

        for i in self.transition:

            total_temp = sum(self.transition[i].values())

            for j in self.transition[i]:
                self.transition[i][j] /= total_temp

        # emission

        for i in self.emission:

            total_temp = sum(self.emission[i].values())

            for j in self.emission[i]:
                self.emission[i][j] /= total_temp


        # TRIPLE TRANSITION

        self.triple = {y: {x: {z : self.small_value for z in self.pos.keys()} for x in self.pos.keys()} for y in self.pos.keys()}
        # self.triple = {y: {x: self.small_value for x in self.pos.keys()} for y in self.pos.keys()}

        # for line in data:
        #     for one,two,three,four in zip(line[1][0:],line[1][2:],line[1][1:],line[1][3:]):
        #         self.triple[one][two] += 1
        #         self.triple[three][four] += 1
        #
        # for k in list(self.triple.keys()):
        #     total_k = sum(self.triple[k].values())
        #     for m in list(self.triple[k].keys()):
        #         self.triple[k][m] /= total_k

        for line in data:
            for one, two, three in zip(line[1][0:len(line[1]) - 2], line[1][1:len(line[1]) - 1], line[1][2:]):
                self.triple[one][two][three] += 1

        for i in self.triple:
            for j in self.triple[i]:

                total_temp = sum(self.triple[i][j].values())

                for k in self.triple[i][j]:
                    self.triple[i][j][k] /= total_temp




        # for line in data:
        #     for one,two,three in zip(line[1][0:len(line[1])-2],line[1][1:len(line[1])-1], line[1][2:]):
        #
        #         try:
        #             self.triple[one][two][three]+=1
        #         except:
        #
        #             try:
        #                 self.triple[one][two][three]=1
        #
        #             except:
        #
        #                 try:
        #                     self.triple[one][two] = {}
        #                     self.triple[one][two][three] = 1
        #                 except:
        #
        #                     self.triple[one] = {}
        #                     self.triple[one][two] = {}
        #                     self.triple[one][two][three] = 1
        # # triple transition total probability
        #
        # for i in self.triple:
        #
        #     for j in self.triple[i]:
        #
        #         total_temp = sum(self.triple[i][j].values())
        #
        #         for k in self.triple[i][j]:
        #             self.triple[i][j][k] /= total_temp








    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    # SIMPLIFIED MODEL
    def simplified(self, sentence):

        parts = list(self.pos.keys())
        final_pos = []
        post = 1

        for word in sentence:

            temp_list=[]

            for s in parts:

                # If word is present

                try:
                    prob = self.pos[s]*self.emission[s][word]
                    temp_list.append(prob)

                # If word is not present, assign a small value for emission

                except:
                    self.emission[s][word]=self.small_value
                    # multiplying the small value assigned with the P(s)
                    prob = self.pos[s] * self.emission[s][word]
                    temp_list.append(prob)

            index = temp_list.index(max(temp_list))

            post*=temp_list[index]

            final_pos.append(parts[index])
        self.simple_post=log(post)
        return final_pos

    # MONTE CARLO MARKOV CHAIN (GIBBS SAMPLING)
    def complex_mcmc(self, sentence):

        parts = list(self.pos.keys())

        initial_state = ['noun']*len(sentence)
        # initial_state = self.simplified(sentence)
        # print(initial_state)

        for i in range(0,100):

            for j in range(0,len(sentence)):

                probability_values = []
                probability_sum = 0

                for k in range(0,len(parts)):
                    t= self.small_value
                    u= self.small_value
                    v= self.small_value
                    wrd = self.small_value

                    part = parts[k]

                    if j==0:
                        t = self.initial[part]
                    elif j==1:
                        t = self.initial[initial_state[j-1]]*self.transition[initial_state[j-1]][part]
                    else:
                        t=1
                        if not(j>=len(sentence)-2):
                            ind = j+2
                        if j==len(sentence)-2:
                            ind = j+1
                            t *= self.transition[initial_state[j]][initial_state[ind]]
                        if j==len(sentence)-1:
                            ind = j

                        while(ind>=1):

                            if ind ==1:
                                t *= self.initial[initial_state[ind-1]]*self.transition[initial_state[ind-1]][initial_state[ind]]
                            else:
                                if ind==j:
                                    t *= self.triple[initial_state[ind-2]][initial_state[ind-1]][part]
                                else:
                                    t *= self.triple[initial_state[ind - 2]][initial_state[ind - 1]][initial_state[ind]]

                            ind -= 1

                    try:
                        wrd = self.emission[part][sentence[j]]
                    except:
                        pass

                    prob = t*wrd
                    probability_sum += prob
                    probability_values.append(prob)


                for m in range(0,len(probability_values)):
                    probability_values[m] = probability_values[m] / probability_sum


                # print(len(parts))
                # print(len(probability_values))
                f_index = int(np.random.choice(len(parts),p = np.array(probability_values)))
                # print(f_index)
                initial_state[j] = parts[f_index]

            self.samples.append(initial_state)

        top_samples = self.samples[len(self.samples)-50:]
        # print(parts)
        # print(probability_values)
        # print(top_samples)
        final=[]

        char_array = np.asanyarray(top_samples)

        for word_count in range(len(sentence)):
            slice = char_array[:,word_count].tolist()
            pos = max(slice, key=slice.count)
            final.append(pos)
        # print(final)

        # print(self.triple['noun']['.']['conj'])
        # print(self.triple['noun']['.']['noun'])
        # sys.exit(0)

        # Calculating posterior
        post=1

        for p in range(len(final)):

            t = self.small_value
            wrd = self.small_value
            u = self.small_value
            v = self.small_value


            if p == 0:
                t=self.initial[final[p]]
            elif p ==1:
                try:
                    t=self.transition[final[p-1]][final[p]]*self.initial[final[p-1]]
                except:
                    pass
            else:
                t = self.triple[final[p - 2]][final[p - 1]][final[p]]


            try:
                wrd = self.emission[final[p]][sentence[p]]
            except:
                pass

            post *= t*wrd


        self.mcmc_post = log(post)


        # print(top_samples)
        # return top_samples[-1]
        return final



    # VITERBI ALGORITHM FOR HIDDEN MARKOV MODEL
    def hmm_viterbi(self, sentence):

        parts = list(self.pos.keys())

        viterbi_table = np.zeros((len(parts),len(sentence)))
        trace = np.zeros((len(parts),len(sentence)),dtype=int)

        for i in range(0,len(sentence)):

            for j in range(0,len(parts)):

                if i==0:

                    part = parts[j]

                    try:
                        viterbi_table[j][i] = log(self.initial[part])+log(self.emission[part][sentence[i]])
                    except:
                        self.emission[part][sentence[i]] = self.small_value
                        viterbi_table[j][i] = log(self.initial[part]) + log(self.emission[part][sentence[i]])

                else:

                    temp_list = []
                    for k in range(0,len(parts)):
                        try:
                            a = self.transition[parts[k]][parts[j]]
                        except:
                            a = self.small_value
                            # self.transition[parts[k]][parts[j]]=self.small_value
                        b = viterbi_table[k][i-1]
                        temp_list.append(log(a)+b)

                    index = temp_list.index(max(temp_list))
                    value = max(temp_list)
                    try:
                        viterbi_table[j][i] = value+log(self.emission[parts[j]][sentence[i]])
                    except:
                        viterbi_table[j][i] = value+log(self.small_value)
                    trace[j][i] = index

        back_last = np.argmax(viterbi_table[:,-1])
        final_pos = [parts[back_last]]

        # for i in range(len(sentence)-1,0,-1):
        for i in range(1,len(sentence)):
            back_last=trace[back_last,-i]
            final_pos.append(parts[back_last])

        # print(len(sentence))
        # print(len(final_pos))
        # print(viterbi_table)
        # print(trace)

        self.viterbi_post = np.max(viterbi_table[:,-1])

        return final_pos[::-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")




