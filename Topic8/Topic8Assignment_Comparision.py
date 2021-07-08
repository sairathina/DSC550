#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Thompson Sampling for Slot Machines

# Importing the libraries
import numpy as np

# Setting conversion rates and the number of samples
# We have 5 slot machines which has some win chances.
conversionRates = [0.15, 0.04, 0.13, 0.11, 0.05]
# from the above it seems the winning chances are 15% for machine 1 etc

# the below is for no of samples and here it is 1000
N = 10000
# the below variable d is for length of conversionrate
d = len(conversionRates)

# Creating the dataset. The variable i will tell us at particular timestep we are won are not for a particular slot machine
X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRates[j]:
            X[i][j] = 1

# Making arrays to count our losses and wins
nPosReward = np.zeros(d)
nNegReward = np.zeros(d)

# Taking our best slot machine through beta distibution and updating its losses and wins
for i in range(N):
    selected = 0
    maxRandom = 0
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j
    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# Showing which slot machine is considered the best
nSelected = nPosReward + nNegReward 
for i in range(d):
    print('Machine number ' + str(i + 1) + ' was selected ' + str(nSelected[i]) + ' times')
print('Conclusion: Best machine is machine number ' + str(np.argmax(nSelected) + 1))

