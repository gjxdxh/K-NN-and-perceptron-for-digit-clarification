import numpy as np
import math
import statistics
import os
import matplotlib as mlb
import matplotlib.pyplot as plt
from scipy import spatial
import numpy.linalg as la

def learning_rate_func_1(x):
    return 0.6 * np.exp(-x * 0.3)
def learning_rate_func_2(x):
    return 0.5/(1+6*x)
with open("trainingimages","r") as f:
    train_data = f.readlines()
with open("traininglabels","r") as f:
    train_label = f.readlines()
train_num = 5000
counter = np.zeros((10,1))
weight_vector = np.zeros((10,14*14+1))
laplace = 0.1
training_value = np.zeros((5000,14*14+1))
for n in range(15):
    accuracy = 0;
    for i in range(train_num):
        curr_label = int(train_label[i])
        counter[curr_label][0] += 1
        if(n == 0):
            for j in range(14):
                for k in range(14):
                    idx = 0;
                    for m in range(2):
                        for x in range(2):
                            if(train_data[28*i+2*j+m][2*k+x] != " "):
                                idx += 0.7
                    training_value[i][14*j+k] = idx
        training_value[i][196] = 1
        for m in range(10):
            judge = np.sign(weight_vector[m].T @ training_value[i])
            if(judge == 1):
                if(m != curr_label):
                    weight_vector[m] -= learning_rate_func_2(n) * training_value[i]
            else:
                if(m == curr_label):
                    weight_vector[m] += learning_rate_func_2(n) * training_value[i]
    for i in range(train_num):
        each_prob = np.zeros(10)
        curr_label = int(train_label[i])
        for m in range(10):
            each_prob[m] = weight_vector[m].T @ training_value[i]
        max_idx = np.argmax(each_prob)
        if(max_idx == curr_label):
            accuracy += 1
    print(str(n+1) + "th epoch accuracy is " + str(accuracy*100/5000) + "%")



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
    temp = np.zeros(14*14+1)
    for j in range(14):
        for k in range(14):
            idx = 0;
            for m in range(2):
                for n in range(2):
                    if(test_data[28*i+2*j+m][2*k+n] != " "):
                        idx += 0.7
            temp[14*j+k] = idx
    temp[196] = 1
    for m in range(10):
        each_prob[m] = weight_vector[m].T @ temp
    max_idx = np.argmax(each_prob)
    if(max_idx != curr_label):
        predict_false += 1

    con_matrix[curr_label][max_idx] += 1

np.set_printoptions(precision=2)
for i in range(10):
    temp = round(con_matrix[i][i] / test_counter[i][0] * 100,1);
    print("Classification rate for " + str(i) + " is " + str(temp) + "%")
print("Overall classification accuracy: " + str((1000 - predict_false)/1000.0 * 100) + "%")
print("Below is the confusion matrix")
print(con_matrix/test_counter)
