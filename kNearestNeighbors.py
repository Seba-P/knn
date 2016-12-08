#!/usr/bin/python

import csv
import random
import math
import operator
 

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	""" Load data from a file and split them into training and testing sets at random (according to 'split' parameter). """
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset) - 1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
 


def calculateDistance(point1, point2, length):
	""" Calculate distance between two data points. """
	distance = 0
	for x in range(length):
		distance += pow((point1[x] - point2[x]), 2)
	return math.sqrt(distance)


def getNeighbors(trainingSet, testPoint, k):
	""" Get k-nearest neighbors of given point. """
	distances = []
	length = len(testPoint)-1
	for x in range(len(trainingSet)):
		dist = calculateDistance(testPoint, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getAnswer(neighbors):
	""" Get point assignment. """
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(iter(classVotes.items()), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
	""" Get accuracy of test set classification. """
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct / float(len(testSet))) * 100.0

