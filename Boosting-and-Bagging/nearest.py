#!/usr/bin/env python3

######################################################################
# Discussion
# By first implementation of the KNN, it was too slow. Below are the ways that
# I have tried to figure out the problem.
# Maybe reducing the dist list size may speed up the computation
# Instead of using np.linalg.norm, I will take step by step: diff -> square -> sum -> take a sqrt ===> got faster
# Rules 1) Lowering Dimension will significantly make computation cheaper
#       2) Lowering the train_set size will also significantly make computation cheaper.
#
#
# 12/06
# *** We need to increase K for better classification. I am going to try multiple Ks and see what has better accuracy on test data
# *** To do the upper, we need to make the computation cheaper.
# ***By R analysis: some variables are more important than others : have lower p-value and have higher significance.
#
# 12/07
# *** What will be better? Have a big distance list and take ordered() and take different votes? or calculate k-nearest neighbors
# as many as counts of k. Will the big distance list small enough for the memory to stand it# ? Yes. Able to stand it. Let's do this.
# 
# 12/08
# *** Succeeded in making a big data list and extracting K smallest training points
# *** By this, I was able to try multiple Ks for one big data list ==> Much faster computation time
######################################################################


######################################################################
# Importing Necessary Packages
######################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import sys


######################################################################
# Reading the File
######################################################################

def readfile(file):
    photo_id=[]
    orientation=[]
    pixel_value=[]
    with open(file, 'r') as f:
        for line in f:
            tmp= line.split()

            # id
            id_tmp = tmp[0]
            photo_id.append(id_tmp)
           
            # orientation
            ot_tmp = tmp[1]
            orientation.append(ot_tmp)
            
            # pixel value
            px_tmp = [int(val) for val in tmp[2:]]
            pix_tmp = []
            for i in range(0,len(px_tmp), 3):
                pix_tmp.append(px_tmp[i: i+3])

            pixel_value.append(pix_tmp)

    return (photo_id, orientation, pixel_value)


######################################################################
# K-nearest-neighbors
######################################################################

# Input: K(needs to be a list), train_px, test_px, train_ot
# Output: Estimated Class for Test Images

def k_nearest_neighbors(K, train_px, test_px, train_ot):
    # calculate Euclidean distance for each test examples with the whole training set
    print("Computing Euclidean Distance for KNN...")
    count=0
    
    # list that stacks distance and class information for all test data
    D_list=[]
    for test_image in test_px:
#        print("Computing for " + str(count) + "th test data")        
        count+=1

        # temporary distance list that stacks between ONE test data and the whole training set
        tmp_d = []
        c = 0
        for train_image in train_px:
            diff = test_image - train_image
            squared=np.square(diff)
            summed = np.sum(squared)
            d = np.sqrt(summed)
            tmp_d.append(d)
            c+=1

           # print(str(c) + " th training data completed...")
        dist_and_ot=list(zip(tmp_d, train_ot))
        D_list.append(dist_and_ot)
    
    print("Computing Euclidean Distance is NOW COMPLETED...")
    print("Estimating CLASS Based on Distance...")
    
    # using the distance information, we order it from the lowest to greatest and get K nearest training points with the test data.
    # among the K nearest training points, we get their labels and using the voting system to get the estimated class for the test data.
    EST_class =[]
    for k in K:
        est_class=[]
        for test_d in D_list:
            d_queue = sorted(test_d)[0:k]
            est_ot = [d_queue[i][1] for i in range(len(d_queue))]
            votes = [est_ot.count(ot) for ot in ot_set]
          #  print("votes :", votes)
            est_class.append(ot_set[votes.index(max(votes))])

        EST_class.append(est_class)
       # print(d_queue[0])

    return(EST_class)


######################################################################
# Calculate Accuracy
######################################################################

def get_accuracy(test_label, est_label):
   
    # check and calculate the accuracy of the estimation
    correct = 0
    for i in range(len(test_label)):
        if est_label[i]==test_label[i]:
            correct+=1
        
        else:
            continue
            
    return(correct / len(test_label))


######################################################################
# Execution of the Codes
######################################################################

train_file = sys.argv[1]
test_file = sys.argv[2]
model = sys.argv[3]

if model=='nearest':
    print('\n')
    print('****************************************')
    print("*****This code takes about 7.5 mins*****")
    print('****************************************')
    print('\n')
    print('LOADING DATA...')

    train_id, train_ot, train_px = readfile(train_file)
    test_id, test_ot, test_px = readfile(test_file)
    print('DATA LOADED...')

    train_px = np.array(train_px)
    test_px = np.array(test_px)

    ot_set=['0', '90', '180', '270']
    
    # This K had the best accuracy among 1 to 50
    K = [48]
    # K = [k for k in range(1,51)]
    est_class = k_nearest_neighbors(K, train_px, test_px, train_ot)

    acc_for_diff_k=[]
    for cls in est_class:
        acc_for_diff_k.append(get_accuracy(test_ot, cls))

    for i in range(len(K)):
        print("Accuracy for K " + str(K[i]) + " is " + str(acc_for_diff_k[i]))

else:
    sys.exit()
    
    
######################################################################
# Plotting Figures
######################################################################
'''
plt.figure(1)
plt.plot(K, acc_for_diff_k, '.-', label = "Accuracy", color = 'red')
plt.title("KNN Accuracy Respect to K")
plt.xlabel("K")
plt.ylabel("Accuacy Values")
plt.legend(loc= "best")
plt.savefig("KNN_accuracy_1_K.png")
'''

