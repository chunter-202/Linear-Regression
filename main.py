   
#############################################
#   COSC 424 Project 2: Linear Regression   #
#   Taylor Chase Hunter                     #
#                                           #
#   This code takes in data from auto-mpg   #
#   And trains a linear regression model on #
#   it.                                     #
#############################################

import csv
import math
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#   Scoring Function    #
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot =sum((Y - mean_y) ** 2)
    ss_res =sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2  

def grad_Descent(data, target, l_Rate):

    ##  Hyper param ##
    max_Itter = 100000

    ##  How long our data is    ##
    num = len(data)

    ##  How man features it is  ##
    features_Num = data.shape[1]

    ##  Create array of that size for weights   ##
    weights = np.zeros(features_Num)

    #   Inizilaize varibles #
    itter = 0
    percent = 0
    b = 0

    while itter < max_Itter:

        ##  Calculate predicted value   ##
        pred = weights*data + b
        pred = pred.sum(axis=1)

        ##  Calculate weights   ##
        for i in range(features_Num):
            m_Step = -(1/num)*sum(data[:,i] * (target-pred))
            weights[i] = weights[i] - m_Step * l_Rate

        ##  Calculate B step    ##
        b_Step = -(1/num)*sum(target-pred)
        b = b - b_Step * l_Rate

        #   Gauges how far along in the training process    #
        if itter % 1000 == 0:
            print("[", percent, "]%")
            percent += 1

        itter += 1

    return weights, b        

#   Read in the data into a pandas structure      #  
data = pd.read_csv(r"C:\Users\sepro\Documents\Cosc Files\425\Project2\auto-mpg.csv")
target = data['mpg'].values

#   Remove target and 'car' feature from training set   #
clean_data = data.drop(['mpg', 'car'], axis=1)

sns.set_theme(style="darkgrid")
size = len(target)
b = 0

standardizedX = clean_data.to_numpy()

#   This was the code used to stndarize the data in testing  #
#scaler = MinMaxScaler(feature_range=(0, 10))
#standardizedX = scaler.fit_transform(clean_data)
#scaler = StandardScaler().fit(standardizedX)
#standardizedX = scaler.transform(standardizedX)

pred = np.zeros(size)
cost = np.zeros(size)

#   Run the algorithem  #
weights, b = grad_Descent(standardizedX, target, .0000001)

#   Find the error rate #
for x in range(size):
    ans =  weights*standardizedX[x]
    ans = ans.sum(axis=0)
    pred[x] = ans + b
    cost[x] = pred[x] - target[x]
    
#   Calculate the r score and print values  #
r2 = r2_score(target, pred)
print("weights: ", weights)
print("Intercept: ", b)
print("Final r2: ", r2)

#   Used for data visulizing    #
#   Left in code for refrence   #
#f, ax = plt.subplots(figsize=(6.5, 6.5))
#sns.despine(f, left=True, bottom=True)
#rw = np.arange(size)
#plt.plot(rw*weights + b)
#sns.scatterplot(y=target, x = rw)
#plt.show()
