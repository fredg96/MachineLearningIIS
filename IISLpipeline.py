import os
import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

labels ={
    "ANGER": 0,
    "DISGUST": 1,
    "FEAR": 2,
    "HAPPY": 3,
    "SADNESS": 4,
    "SURPRISE": 5,
    }

def readTestFile():
    currLoc = os.getcwd()
    path = currLoc +"/Testing"
    files =[]
    data = []
    labels =[]
    names =[]
    for file in os.listdir(path):
        if file.endswith(".lm3"):  # scan for lm3 files
            files.append(str(file))
    for file in files:
        lm3Data = parseFile(path + "/" + file)  # Open each file
        if lm3Data:  # Add the data as a list, the labels and the names
            data.append(lm3Data)
            labels.append(file.split("_")[2])
            names.append(file)
    return data, labels, names

def parseFile(filePath):
    """
    :param filePath: location of datafiles
    :return: list with landmarks, comment out landmarks and they won't be read
    """
    landmarkNames = { #Remove those which won't be tracked for real life one
        "Outer left eyebrow",
        "Middle left eyebrow",
        "Inner left eyebrow",
        "Inner right eyebrow",
        "Middle right eyebrow",
        "Outer right eyebrow",
        "Outer left eye corner",
        "Inner left eye corner",
        "Inner right eye corner",
        "Outer right eye corner",
        "Nose saddle left",
        "Nose saddle right",
        "Left nose peak",
        "Nose tip",
        "Right nose peak",
        "Left mouth corner",
        "Upper lip outer middle",
        "Right mouth corner",
        "Upper lip inner middle",
        "Lower lip inner middle",
        "Lower lip outer middle",
        "Chin middle",
    }
    with open(filePath) as f:
        lines = f.readlines()
    landmarks = []
    for i in range(4, len(lines), 2):
        if lines[i - 1].rstrip() in landmarkNames:
            landmark = [float(j) for j in lines[i].split()]
            landmarks.append(landmark)
    return landmarks

def calcAllDistanceFeatures(data):
    dataPoints = len(data)
    feats = np.zeros((dataPoints,22*22))

    for i in range(dataPoints):
        index = 0
        for j in range(21):
            for k in range(21):
                if(j != k):
                    feats[i,index] = calcDist(data[i][j],data[i][k])
                    index = index+1
    return feats

def calcDist(pointA,pointB):
    return distance.euclidean(pointA,pointB)

def predConf(classifier, data):
    features = np.array([data])
    features = calcAllDistanceFeatures(data)
    prob = classifier.predict_proba(features)[0]
    features = preprocessing.normalize(features)
    features = features.reshape(features.shape[0], -1)
    index = np.where(prob == np.amax(prob))[0]
    return index, prob

classifier = joblib.load('randomForest.joblib')
data, label, name = readTestFile()
index, probs = predConf(classifier,data)
print(index)
print(probs)