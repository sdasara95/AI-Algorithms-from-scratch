#!usr/bin/env python3

import numpy as np
import os

os.getcwd()
os.chdir('C:\\Users\\swlee\\OneDrive\\Desktop')

def readfile(file):
    print("Loading data...")
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
            
        print("DATA LOADED")
    return (photo_id, orientation, pixel_value)


train_id, train_ot, train_px=readfile('train-data.txt')
test_id, test_ot, test_px=readfile('test-data.txt')

#for i in range(0,len(train_ot),4):
#    print(train_ot[i]+ "\n")

train_px = np.array(train_px)
test_px = np.array(test_px)


# Input: K, train_px, test_px, train_ot, test ot (when accuracy happens inside this) |Output: Estimated Class for Test Images
# for K, since we have 4 orientation, K=4 seems ideal
######################################################################
# Discussion
# By first implementation of the KNN, it was too slow. Below are the ways that
# I have tried to figure out the problem.
# Maybe reducing the dist list size may speed up the computation
# Instead of using np.linalg.norm, I will take step by step: diff -> square -> sum -> take a sqrt
# Rules 1) Lowering Dimension will significantly make computation cheaper
#       2) Lowering the train_set size will also significantly make computation cheaper.
#
#
# 12/06
# *** We need to increase K for better classification. I am going to try multiple Ks and see what has better accuracy on test data
# *** To do the upper, we need to make the computation cheaper.
# ***By R analysis: some variables are more important than others : have lower p-value and have higher significance.
######################################################################


ot_set=['0', '90', '180', '270']
def k_nearest_neighbors(K, train_px, test_px, train_ot, test_ot):
    print("Computing Euclidean Distance for KNN...")
    est_class=[]
    count=0
    for test_image in test_px:
        print("Computing for " + str(count) + "th test data")        
        count+=1
        # initialize dist=0 
        dist = []
        c = 0
        for train_image in train_px:
            diff = test_image - train_image
            squared=np.square(diff)
            summed = np.sum(squared)
            d = np.sqrt(summed)
            dist.append(d)    
            if len(dist) == K:
                dist.remove(min(dist))
            c+=1
           # print(str(c) + " th training data completed...")
        dist_and_ot=list(zip(dist, train_ot))
        
        # sorting based on the distance
        topKlowest=sorted(dist_and_ot)[0:K]
        topK_ot = [topKlowest[i][1] for i in range(len(topKlowest))]
        
        votes = [topK_ot.count(ot) for ot in ot_set]
        est_class.append(ot_set[votes.index(max(votes))])
                                
    print("KNN Completed. Estimated Class Generated...")                            
    return(est_class)

def get_accuracy(test_label, est_label):
    # check and calculate the accuracy of the estimation
    correct = 0
    for i in range(len(test_label)):
        if est_label[i]==test_label[i]:
            correct+=1
        
        else:
            continue
            
            
    return(correct / len(test_label))
K=5
for k in range(1,K):
    est_class = k_nearest_neighbors(k, train_px, test_px, train_ot, test_ot)
    print("Accuracy for K " + str(k) + " is " + str(get_accuracy(test_ot, est_class)))
