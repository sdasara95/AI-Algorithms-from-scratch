#!/usr/bin/env python3

import os
import random
import numpy as np
from collections import Counter
import json
import sys

# REPORT
#
# UPLOADED PDF FILES FOR INDIVIDUAL ALGORITHM IN GITHUB
#
# Best Algorithm:
#   Our best algorithm is KNN.
#

# Adaboost Algorithm
def trainAdaboost(train,model_file):
    print("TRAINING STARTED....This takes around 2 minute")
    raw = []
    train_image_name = []
    train_pixels = []

    with open(train) as f:
        for line in f:
            raw = line.split()
            train_image_name.append(raw[0])
            train_pixels.append(np.array([int(i) for i in raw[1:194]]))

    train_pixels = np.array(train_pixels)
    indices = []
    for i in range(1, 192):
        for j in range(i, 192):
            indices.append((i, j))
    train_parameter = {"stump": [], "angle": [], "a": []}

    random.seed(0)
    stumps = random.sample(indices, 500)

    for stump in stumps:

        weights = np.array([float(1 / len(train_pixels))] * len(train_pixels))

        error = 0
        count = 0

        positive = train_pixels[train_pixels[:, stump[0]] >= train_pixels[:, stump[1]]][:, 0]
        negative = train_pixels[train_pixels[:, stump[0]] < train_pixels[:, stump[1]]][:, 0]

        if len(positive) > 0:
            positive_class = Counter(positive).most_common()[0][0]
        if len(negative) > 0:
            negative_class = Counter(negative).most_common()[0][0]

        temp_class = []

        sign = train_pixels[:, stump[0]] >= train_pixels[:, stump[1]]

        for i2 in range(len(sign)):
            if sign[i2] == True:
                temp_class.append(positive_class)
            else:
                temp_class.append(negative_class)
            if temp_class[i2] != train_pixels[i2][0]:
                count += 1

        error = count / len(train_pixels)

        if error > .75:
            continue
        a = np.log((1 - error) / error) + np.log(3)

        for i5 in range(len(train_pixels)):
            if temp_class[i5] != train_pixels[i5][0]:
                weights[i5] = weights[i5] * np.exp(a)

        train_parameter["stump"].append((int(stump[0]), int(stump[1])))
        train_parameter["angle"].append((int(positive_class), int(negative_class)))
        train_parameter["a"].append(float(a))

    file = open(model_file, 'w')
    file.write(json.dumps(train_parameter))
    file.close()
    print("TRAINING IS DONE")


def testAdaboost(test,model_file):
    print("TESTING STARTED")
    test_raw = []
    test_image_name = []
    test_pixels = []

    with open(test) as f:
        for line in f:
            test_raw = line.split()
            test_image_name.append(test_raw[0])
            test_pixels.append(np.array([int(i) for i in test_raw[1:194]]))

    test_pixels = np.array(test_pixels)
    file = open(model_file, 'r');
    parameters = json.load(file)
    ans = []
    count1 = 0
    output = open("output_adaboost.txt", "w")
    for i in range(len(test_pixels)):
        temp_ans = {0: 0, 90: 0, 180: 0, 270: 0}
        for j in range(len(list(parameters.values())[0])):
            index1, index2 = list(parameters.values())[0][j]
            angle1, angle2 = list(parameters.values())[1][j]
            w = list(parameters.values())[2][j]
            if test_pixels[i][index1] >= test_pixels[i][index2]:
                temp_ans[angle1] += w
            else:
                temp_ans[angle2] += w
        ans.append(test_image_name[i] + " " + str(list(temp_ans.keys())[list(temp_ans.values()).index(max(list(temp_ans.values())))]))
        if int(ans[i].split()[1]) == test_pixels[i][0]:
            count1 += 1
        output.writelines("%s\n" % ans[i])
    print("THE ACCURACY IS")
    print(count1 / len(test_pixels))
    output.close()
################################################################################
# FOREST
# I have referred to this link for the implementation of the decision tree:
# https://www.cs.purdue.edu/homes/ribeirob/courses/Spring2018/notes/decision_tree.html

def trainForest(train,model_file):
    global tree_list,train_ratio,train_len,train_size,max_depth,min_sample
    print('TRAINING HAS STARTED FOR DECISION FOREST!! This takes around 21 minutes')
    def read(fname):
        X = np.loadtxt(fname, usecols=range(2,194))
        Y = np.loadtxt(fname,usecols=1)
        return X, Y
    global X,Y
    X,Y = read(train)
    Xy = np.column_stack((X,Y))
    global threshold_values
    threshold_values = np.median(X,axis=0)
    
    def random_indices(n):
        a = np.arange(len(Xy))
        np.random.shuffle(a)
        return a[0:n]
        
    def gini_score(groups,classes):
        n_samples = sum([len(group) for group in groups])
        gini = 0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            #print(size)
            for class_val in classes:
                #print(group.shape)
                p = (group[:,-1] == class_val).sum() / size
                score += p * p
            gini += (1.0 - score) * (size / n_samples)
            #print(gini)
        return gini
    
    def split(feat,data):
        val = threshold_values[feat]
        left, right = np.array([]).reshape(0,len(data[0])), np.array([]).reshape(0,len(data[0]))
        for row in data:
            if row[feat]<=val:
                left = np.vstack((left,row))
            else:
                right = np.vstack((right,row))
        return left,right
    
    def best_split(data):
        classes = list(set(row[-1] for row in data))
        b_feat, b_value, b_score, b_groups = 999,999,999,None


        for feature in range(data.shape[1]-1):
            groups = split(feature,data)
            gini = gini_score(groups,classes)
            value = threshold_values[feature]
            if gini<b_score:
                b_feat, b_score, b_groups, b_value = feature, gini, groups, value

        return {'index':b_feat,'value':b_value,'groups':b_groups}
        
    def terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    def branch(node, max_depth, min_sample, depth):
        left, right = node['groups']
        del(node['groups'])
    
        if not isinstance(left,np.ndarray) or not isinstance(right,np.ndarray):
            node['left'] = node['right'] = terminal_node(np.concatenate(left,right))
            return

        if depth >= max_depth:
            node['left'],node['right'] = terminal(left),terminal(right)
            return

        if len(left)<=min_sample:
            node['left']=terminal(left)
        else:
            node['left']=best_split(left)
            branch(node['left'],max_depth,min_sample,depth+1)

        if len(right)<=min_sample:
            node['right']=terminal(right)
        else:
            node['right'] = best_split(right)
            branch(node['right'],max_depth,min_sample,depth+1)
        
    def tree(data,max_depth,min_sample):
        root = best_split(data)
        branch(root,max_depth,min_sample,1)
        return root
        
    def predict(model_tree,data):
        if data[model_tree['index']] < model_tree['value']:
            if isinstance(model_tree['left'],dict):
                return predict(model_tree['left'],data)
            else:
                return model_tree['left']
        else:
            if isinstance(model_tree['right'],dict):
                return predict(model_tree['right'],data)
            else:
                return model_tree['right']
                
    tree_list = []
    num_trees = 30
    train_ratio = 0.03
    train_len = len(Xy)
    train_size = int(train_ratio*train_len)
    max_depth = 5
    min_sample = 50
    
    for i in range(num_trees):
        indices = random_indices(train_size)
        new_tree = tree(Xy[indices],max_depth,min_sample)
        tree_list.append(new_tree)
    
    model_file1 =  model_file
    file = open(model_file1, 'w')
    file.write(json.dumps(tree_list))
    file.close()
    def accuracy(tX,tY,tXy,tree_list):
        correct = 0
        for i in range(len(tX)):
            row = tX[i]
            class_counts = {0:0,90:0,180:0,270:0}
            for j in range(len(tree_list)):
                prediction = int(predict(tree_list[j],row))
                class_counts[prediction] += 1
            labels = list(class_counts.values())
            key_index = labels.index(max(labels))
            final = list(class_counts.keys())[key_index]
            if final==tY[i]:
                correct+=1
        

        ac = correct/len(tY)
        return ac
        
    print('TRAINING HAS ENDED FOR DECISION FOREST')
    accu = accuracy(X,Y,Xy,tree_list)
    print('Train Accuracy is :',accu)
    
    
    
def testForest(test,model_file):
    print('TESTING HAS STARTED FOR DECISION FOREST!!')
    file1 = open(model_file,'r')
    parameters = json.load(file1)
    file_names = np.loadtxt(test ,dtype='str',usecols=0).tolist()
    
    def read(fname):
        X = np.loadtxt(fname, usecols=range(2,194))
        Y = np.loadtxt(fname,usecols=1)
        return X, Y
    global X,Y
    X,Y = read(test)
    Xy = np.column_stack((X,Y))
    file = open(model_file, 'r');
    
    def predict(model_tree,data):
        if data[model_tree['index']] < model_tree['value']:
            if isinstance(model_tree['left'],dict):
                return predict(model_tree['left'],data)
            else:
                return model_tree['left']
        else:
            if isinstance(model_tree['right'],dict):
                return predict(model_tree['right'],data)
            else:
                return model_tree['right']
    
    def accuracy(tX,tY,tXy,tree_list):
        correct = 0
        for i in range(len(tX)):
            row = tX[i]
            class_counts = {0:0,90:0,180:0,270:0}
            for j in range(len(tree_list)):
                prediction = int(predict(tree_list[j],row))
                class_counts[prediction] += 1
            labels = list(class_counts.values())
            key_index = labels.index(max(labels))
            final = list(class_counts.keys())[key_index]
            if final==tY[i]:
                correct+=1

        accuracy = correct/len(tY)
        return accuracy
    
    accu = accuracy(X,Y,Xy,parameters)
    
    print(' Test Accuracy is : ',accu)
    
    def final_prediction(row,tree_list):
        class_counts = {0:0,90:0,180:0,270:0}
        for j in range(len(tree_list)):
            prediction = int(predict(tree_list[j],row))
            class_counts[prediction] += 1
        labels = list(class_counts.values())
        key_index = labels.index(max(labels))
        final = list(class_counts.keys())[key_index]
        return final
    
    predictions = []
    for row in X:
        predictions.append(final_prediction(row,parameters))
       
    output = open("output_forest.txt", "w")
    for i in range(len(X)):
        term=str(file_names[i])+' '+str(predictions[i])
        output.writelines("%s\n" % term)
    
    print('Testing done')
    
    
#################################################################################
#<<<<<<< HEAD
def trainNearest(train,model_file):
    print("Training Phase : Since KNN does not have training phase, this creates model_file.txt, which copies the examples from the training_file.txt")

    train = open(train, 'r')
    model = open(model_file, 'w')

    for line in train:
        model.writelines(line)

    train.close()
    model.close()
#    pass
def testNearest(test, model_file):
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
    # Reading the File
    ######################################################################

    def readfile(file):
        photo_id = []
        orientation = []
        pixel_value = []
        with open(file, 'r') as f:
            for line in f:
                tmp = line.split()

                # id
                id_tmp = tmp[0]
                photo_id.append(id_tmp)

                # orientation
                ot_tmp = tmp[1]
                orientation.append(ot_tmp)

                # pixel value
                px_tmp = [int(val) for val in tmp[2:]]
                pix_tmp = []
                for i in range(0, len(px_tmp), 3):
                    pix_tmp.append(px_tmp[i: i + 3])

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
        #    count=0

        # list that stacks distance and class information for all test data
        D_list = []
        for test_image in test_px:
            #       print("Computing for " + str(count) + "th test data")
            #      count+=1

            # temporary distance list that stacks between ONE test data and the whole training set
            tmp_d = []
            #     c = 0
            for train_image in train_px:
                diff = test_image - train_image
                squared = np.square(diff)
                summed = np.sum(squared)
                d = np.sqrt(summed)
                tmp_d.append(d)
            #        c+=1

            # print(str(c) + " th training data completed...")
            dist_and_ot = list(zip(tmp_d, train_ot))
            D_list.append(dist_and_ot)

        print("Computing Euclidean Distance is NOW COMPLETED...")
        print("Estimating CLASS Based on Distance...")

        # using the distance information, we order it from the lowest to greatest and get K nearest training points with the test data.
        # among the K nearest training points, we get their labels and using the voting system to get the estimated class for the test data.
        EST_class = []
        for k in K:
            est_class = []
            for test_d in D_list:
                d_queue = sorted(test_d)[0:k]
                est_ot = [d_queue[i][1] for i in range(len(d_queue))]
                votes = [est_ot.count(ot) for ot in ot_set]
                #  print("votes :", votes)
                est_class.append(ot_set[votes.index(max(votes))])

            EST_class.append(est_class)
        # print(d_queue[0])

        return (EST_class)

    ######################################################################
    # Calculate Accuracy
    ######################################################################

    def get_accuracy(test_label, est_label):

        # check and calculate the accuracy of the estimation
        correct = 0
        for i in range(len(test_label)):
            if est_label[i] == test_label[i]:
                correct += 1

            else:
                continue

        return (correct / len(test_label))

    ######################################################################
    # Execution of the Codes
    ######################################################################
    #
    # train_file = sys.argv[1]
    # test_file = sys.argv[2]
    # model = sys.argv[3]

    # if model=='nearest':
    print('\n')
    print('****************************************')
    print("*****This code takes about 7.5 mins*****")
    print('****************************************')
    print('\n')
    print('LOADING DATA...')

    train_id, train_ot, train_px = readfile(model_file)
    test_id, test_ot, test_px = readfile(train_test_file)
    print('DATA LOADED...')

    train_px = np.array(train_px)
    test_px = np.array(test_px)

    ot_set = ['0', '90', '180', '270']

    # This K had the best accuracy among 1 to 50
    K = [48]
    # K = [k for k in range(1,51)]
    est_class = k_nearest_neighbors(K, train_px, test_px, train_ot)
    # for i in range(len(test_id)):
    #    print(test_id[i] + ' ' + est_class[0][i])
    # Outputting the test id and estimated class on txt file
    with open("output_nearest.txt", 'w') as outputfile:
        for i in range(len(test_id)):
            outputfile.writelines(test_id[i] + ' ' + est_class[0][i] + '\n')

    acc_for_diff_k = []
    for cls in est_class:
        acc_for_diff_k.append(get_accuracy(test_ot, cls))

    for i in range(len(K)):
        print("Accuracy for K = " + str(K[i]) + " is " + str(acc_for_diff_k[i]))

    # else:
    #    sys.exit()

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


# =======
# def trainNearest(train,model_file):
#     pass
# def testNearest(test,model_file):
# pass
# >>>>>>> 1735cf7b2853f95589d2b13f62c05d2f4d61a28d

if __name__ == '__main__':

    option = sys.argv[1]
    train_test_file = sys.argv[2]
    model_file = sys.argv[3]
    model_used = sys.argv[4]


    if(option == 'train'):

        if(model_used == 'adaboost'):
            trainAdaboost(train_test_file,model_file)

        elif(model_used == 'nearest'):
#<<<<<<< HEAD
#=======
            trainNearest(train_test_file,model_file)
#>>>>>>> 1735cf7b2853f95589d2b13f62c05d2f4d61a28d

        elif(model_used == 'forest'):
            trainForest(train_test_file,model_file)


        elif(model_used == 'best'):
            trainNearest(train_test_file, model_file)

    elif(option == 'test'):

        if (model_used == 'adaboost'):
            testAdaboost(train_test_file, model_file)

        elif (model_used == 'nearest'):
            testNearest(train_test_file, model_file)
#<<<<<<< HEAD
            
#=======

#>>>>>>> 1735cf7b2853f95589d2b13f62c05d2f4d61a28d
        elif (model_used == 'forest'):
            testForest(train_test_file, model_file)


        elif (model_used == 'best'):
#<<<<<<< HEAD
            testNearest(train_test_file, model_file)
#=======
#            testNearest(train_test_file, model_file)
#>>>>>>> 1735cf7b2853f95589d2b13f62c05d2f4d61a28d
