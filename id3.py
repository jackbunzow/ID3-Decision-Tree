#!/usr/bin/env python3

# Jack Bunzow
# Dr. Phillips
# 11/14/19
# Project 3
# This program reads in two data files from the command line. First file being the training data
# which is then used to build an ID3 decision tree. Second file is used to test the tree.
# The number of correct classifications from the testing data is output
import sys, math
import numpy as np

# node class
class node():
    def __init__(self):
        self.terminal = False
        self.classification = -1
        self.attribute = -1
        self.value = -1.0
        self.left = None
        self.right = None

# get information gain
def informationGain(head, last, column, data, indices, numAttr, numClass):
    info = 0

    for i in range(0, numClass):
        count = 0
        for j in range(head, last):
            if data[indices[j][column], numAttr] == i:
                count += 1

        if last != 0 and count != 0:
            p = count / last
            I = -1 * p * np.log2(p)
        else:
            I = 0
        
        info += I

    return info

# get the split attribute and split average
def getSplit(last, info, indices, data, numAttr, numClass):
    minimum = info
    splitAvg = 0.0
    head = 0
    
    splitAttribute = data.shape[1]-1
    for i in range(data.shape[1]-1):
        for j in range(head, last-1):
            if data[indices[j, i], i] != data[indices[j+1, i], i]:
                E = ((j+1)/last) * informationGain(head, j+1, i, data, indices, numAttr, numClass) \
                    + ((last - j+1) / last) * informationGain(j+1, last, i, data, indices, numAttr, numClass)

                if E < minimum: #minimum entropy = max gain
                    minimum = E
                    splitAttribute = i
                    splitAvg = (data[indices[j, i], i] + data[indices[j+1, i], i]) / 2

    return splitAttribute, splitAvg

# build the tree from training data
def build(data, numClass, numAttr, indices):
    head = 0
    last = data.shape[0]
    indices = np.argsort(data, axis=0)
    info = informationGain(0, last, 0, data, indices, numAttr, numClass)

    if info == 0: # if terminal node
        newNode = node()
        newNode.terminal = True
        newNode.classification = data[0, numAttr]
        newNode.attribute = -1
        newNode.value = -1.0
        newNode.left = None
        newNode.right = None
        return newNode
    else: # make children nodes
        newNode = node()
        newNode.attribute, newNode.value = getSplit(last, info, indices, data, numAttr, numClass)
        
        if newNode.value == 0: # if terminal node
            newNode = node()
            newNode.terminal = True
            newNode.attribute = -1
            newNode.value = -1.0
            newNode.left = None
            newNode.right = None

            # unnecessarily long way to get the class label for equal number of each class label
            # left in the data. Picks the smallest number class
            blah, count= np.unique(data[:, data.shape[1]-1], return_counts=True)
            idk = np.unique(count)
            if len(idk) == 1:
                newNode.classification = blah[0]
            else: # majority vote class label
                index = np.where(count == np.amax(count))
                pointlessVariable = blah[index[0]]
                newNode.classification = pointlessVariable[0]

            return newNode

        newNode.terminal = False
        newNode.classification = -1

        # get row to split at
        temp = data[indices[:, newNode.attribute][0:], newNode.attribute].copy()
        count = 0
        for i in range(len(temp)):
            if temp[i] < newNode.value:
                count += 1

        # split the data at the row found
        # anything above the row goes to the leftData
        # anything below goes to the rightData
        leftData = data[indices[:, newNode.attribute][:count], :].copy()
        rightData = data[indices[:, newNode.attribute][count:], :].copy()

        # create indices lists corresponding to the left and right data
        leftIndicies = indices[:,:][:count].copy()
        rightIndicies = indices[:,:][count:].copy()

        # create left and right children
        newNode.left = build(leftData, numClass, numAttr, leftIndicies)
        newNode.right = build(rightData, numClass, numAttr, rightIndicies)

        return newNode

# test the data against the tree
def test(testingData, training):
    correct = 0

    if testingData.ndim > 1: # if more than one row of testing data
        for i in range(testingData.shape[0]):
            current = training
            while current.value != -1:
                if testingData[i, current.attribute] < current.value: # go left
                    current = current.left
                else: # go right
                    current = current.right

            if testingData[i, testingData.shape[1]-1] == current.classification:
                correct += 1

    else: # if only one row of testing data
        current = training
        while current.value != -1:
            if testingData[current.attribute] < current.value: # go left
                current = current.left
            else: # go right
                current = current.right
        
        if testingData[len(testingData)-1] == current.classification:
                correct += 1

    return correct



def main():
    # read in files
    trainingFile = sys.argv[1]
    trainingData = np.loadtxt(trainingFile)
    testingFile = sys.argv[2]
    testingData = np.loadtxt(testingFile)

    if len(trainingData.shape) < 2:
        trainingData = np.array([trainingData])
    if len(testingData.shape) < 2:
        testingData = np.array([testingData])


    labels = np.unique(trainingData[:, trainingData.shape[1]-1]) # all unique class labels
    numClass = len(labels) # number of classes
    numAttr = trainingData.shape[1] - 1 # number of attributes (columns)

    indices = np.argsort(trainingData, axis=0) # locations of all the data in sorted order

    root = build(trainingData, numClass, numAttr, indices) # build the tree
    results = test(testingData, root) # test the testing data
    print(results) # print the number of correct classifications

if __name__ == "__main__":
    main()
