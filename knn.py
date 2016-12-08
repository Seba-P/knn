#!/usr/bin/python

import kNearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier

# Number of nearest neighbors parameter
K = 2

# Percentage of data used as a training set
split = 0.67

# Data training and testing sets
trainingSet = []
testSet = []

# Results of data classification for both algorithms
ourPredictions = []
sklPredictions = []

# Prepare both sets from database
kNN.loadDataset('iris.data', split, trainingSet, testSet)
print('Train set: ' + repr(len(trainingSet)))
print('Test set: ' + repr(len(testSet)))

# Fit training data values with corresponding labels
trainingX = [trainingSet[x][0:4] for x in range(len(trainingSet))]
trainingY = [trainingSet[x][4] for x in range(len(trainingSet))]
sklNeigh = KNeighborsClassifier(n_neighbors=K)
sklNeigh.fit(trainingX, trainingY)

# Perform data classification
for x in range(len(testSet)):
	sklResult = sklNeigh.predict([testSet[x][0:4]])	
	sklPredictions.append(sklResult)

	ourNeighbors = kNN.getNeighbors(trainingSet, testSet[x], K)	
	ourResult = kNN.getAnswear(ourNeighbors)
	ourPredictions.append(ourResult)

	print((repr(x) + ' ourResult = ' + repr(ourResult) + ', sklResult = ' + repr(sklResult[0]) + ', actual = ' + repr(testSet[x][-1])))

# Calculate accuracy of both algorithms
ourAccuracy = kNN.getAccuracy(testSet, ourPredictions)
sklAccuracy = kNN.getAccuracy(testSet, sklPredictions)
print(('Our accuracy: ' + repr(ourAccuracy) + '%'))
print(('Skl accuracy: ' + repr(sklAccuracy) + '%'))
	



