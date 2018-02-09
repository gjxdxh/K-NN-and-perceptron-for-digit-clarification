import numpy as np
import math
import statistics
import os
import matplotlib as mlb
import matplotlib.pyplot as plt
from scipy import spatial
import numpy.linalg as la
import operator
import time

def similarity_func(a,b):
    result = a - b
    return la.norm(result)

if __name__ == "__main__":
    start_time = time.process_time()
    k_nn = 4;
    with open("trainingimages","r") as f:
        train_data = f.readlines()
    with open("traininglabels","r") as f:
        train_label = f.readlines()
    train_num = 5000
    counter = np.zeros((10,1))
    training_value = np.zeros((5000,28*28))
    for i in range(train_num):
        curr_label = int(train_label[i])
        counter[curr_label][0] += 1
        for j in range(28):
            for k in range(28):
                if train_data[28*i+j][k] == " ":
                    training_value[i][28*j+k]= 0
                else:
                    training_value[i][28*j+k] = 1
    with open("testimages","r") as f:
        test_data = f.readlines()
    with open("testlabels","r") as f:
        test_label = f.readlines()
    predict_false = 0.0;
    test_num = 1000;
    test_counter = np.zeros((10,1))
    con_matrix = np.zeros((10,10))
    for i in range(test_num):
        curr_label = int(test_label[i])
        test_counter[curr_label][0] += 1
        each_prob = np.zeros(10)
        temp = np.zeros(28*28)
        for j in range(28):
            for k in range(28):
                if test_data[28*i+j][k] == " ":
                    temp[28*j+k] = 0
                else:
                    temp[28*j+k] = 1
        distance = []
        for m in range(5000):
            temp_dis = similarity_func(training_value[m],temp)
            distance.append((int(train_label[m]),temp_dis))
        distance.sort(key = operator.itemgetter(1))
        neighbors = []
        for x in range(k_nn):
    	       neighbors.append(distance[x][0])
        for j in range(len(neighbors)):
            each_prob[neighbors[j]] += 1
        max_idx = np.argmax(each_prob)
        if(max_idx != curr_label):
            predict_false += 1

        con_matrix[curr_label][max_idx] += 1
    end_time = time.process_time()
    print("Total time is " + str(end_time-start_time) + "s")
    np.set_printoptions(precision=2)
    for i in range(10):
        temp = round(con_matrix[i][i] / test_counter[i][0] * 100,1);
        print("Classification rate for " + str(i) + " is " + str(temp) + "%")
    print("Overall classification accuracy: " + str((1000 - predict_false)/1000.0 * 100) + "%")
    print("Below is the confusion matrix")
    print(con_matrix/test_counter)
