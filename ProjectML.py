import os
import itertools

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn import manifold, neighbors, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix


labels ={
    "ANGER": 0,
    "DISGUST": 1,
    "FEAR": 2,
    "HAPPY": 3,
    "SADNESS": 4,
    "SURPRISE": 5,
    }

keys = labels.keys()

class classData:
    def __init__(self):
        data, label, name = readFile()
        self.data = np.array(data)
        self.features = self.data
        self.label = [labels[i] for i in label]
        self.name = name
        self.classes = 6
        self.n_files = 0

class classData2:
    def __init__(self):
        self.classes = 6
        self.n_files = 0
        self.features =[]
        self.label =[]
        self.validData = []
        self.validLabels =[]

# <editor-fold desc="Read/Parse lm3File">
def readTestFile():
    currLoc = os.getcwd()
    path = currLoc +"/dataBosphoruslm3/Testing"
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

def readFile():
    currLoc = os.getcwd() # Current path
    path = currLoc +"/dataBosphoruslm3"  # complete path to file with dataset
    files = []
    data = []
    labels = []
    names = []
    for file in os.listdir(path):
        if file.endswith(".lm3"):  # scan for lm3 files
            files.append(str(file))
    for file in files:
        lm3Data = parseFile(path+"/"+file)  # Open each file
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
# </editor-fold>

# <editor-fold desc="Feature engineering">
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

def scaledFeatEng(data):
    numData = len(data)
    numFeatures = 10
    feats = np.zeros((numData, numFeatures))
    for i in range(numData):
        scale = np.mean(calcFaceSize(data[i]))
        feats[i, 0] = calcScaledDist(data[i][2], data[i][3], scale) #Eyebrow separation
        feats[1,1] = calcScaledDist(data[i][14],data[i][13],scale) #Right nose wing to middle
        feats[i,2] = calcScaledDist(data[i][16], data[i][20],scale) #Outer lip separation
        feats[i,3] = calcScaledDist(data[i][18],data[i][19],scale) # Inner lip separation
        feats[i,4] = calcScaledDist(data[i][6],data[i][7],scale) #Zoom using size of eye
        feats[i,5] = calcScaledDist(data[i][4],data[i][8],scale) #right eye to eyebrow
        feats[i,6] = calcScaledDist(data[i][1],data[i][7],scale) #left eye to eyebrow
        feats[i,7] = calcScaledDist(data[i][12],data[i][13],scale) #left nose wing to middle
        feats[i,8] = calcScaledDist(data[i][17],data[i][13],scale) # right mouth corner to nose
        feats[i,9] = calcScaledDist(data[i][15],data[i][13],scale) # left mouth corner to nose
        #feats[i, 10] = calcScaledDist(data[i][17], data[i][21], scale)  # right mouth corner to chin
        #feats[i, 11] = calcScaledDist(data[i][15], data[i][21], scale)  # left mouth corner to chin
    return feats

def featEng(data):
    numData = len(data)
    numFeatures = 10
    feats = np.zeros((numData, numFeatures))
    for i in range(numData):
        scale = np.mean(calcFaceSize(data[i]))
        feats[i, 0] = calcScaledDist(data[i][2], data[i][3], scale) #Eyebrow separation
        feats[1,1] = calcScaledDist(data[i][14],data[i][13],scale) #Right nose wing to middle
        feats[i,2] = calcScaledDist(data[i][16], data[i][20],scale) #Outer lip separation
        feats[i,3] = calcScaledDist(data[i][18],data[i][19],scale) # Inner lip separation
        feats[i,4] = calcScaledDist(data[i][6],data[i][8],scale) #Zoom using size of eye
        feats[i,5] = calcScaledDist(data[i][4],data[i][8],scale) #right eye to eyebrow
        feats[i,6] = calcScaledDist(data[i][1],data[i][7],scale) #left eye to eyebrow
        feats[i,7] = calcScaledDist(data[i][12],data[i][13],scale) #left nose wing to middle
        feats[i,8] = calcScaledDist(data[i][17],data[i][13],scale) # right mouth corner to nose
        feats[i,9] = calcScaledDist(data[i][15],data[i][13],scale) # left mouth corner to nose
        #feats[i, 10] = calcScaledDist(data[i][18], data[i][21], scale)  # right mouth corner to chin
        #feats[i, 11] = calcScaledDist(data[i][15], data[i][21], scale)  # left mouth corner to chin
    return feats

def calcFaceSize(points):
    """"
    Calculates the distance between each feature and the nose, feature 14
    """
    dist = []
    for i in range(len(points)):
        if i != 14:
            dist.append(distance.euclidean(points[14],points[i]))
    return dist

def calcScaledDist(pointA,pointB,scale):
    return distance.euclidean(pointA,pointB)/scale

def calcDist(pointA,pointB):
    return distance.euclidean(pointA,pointB)
# </editor-fold>

# split data
def holdOut(dataClass, percentSplit):
    trainData, testData, trainLabels, expectedLabels = train_test_split(dataClass.features, dataClass.label,
                                                                       test_size=(1.0 - percentSplit), random_state=0)
    return trainData,testData,trainLabels,expectedLabels

# <editor-fold desc="Data Augmentation">
def augment(origData, origLabel, runs, strength):
    labelToAppend = origLabel
    dataToPermutate = origData
    for i in range(runs):
        origLabel = origLabel + labelToAppend
        origData = np.vstack((origData, plusNoise(dataToPermutate, strength)))
        origLabel = origLabel + labelToAppend
        origData = np.vstack((origData, negNoise(dataToPermutate, strength)))
    return origData, origLabel

def plusNoise(inData, strength):
    data = inData
    for i in range(len(inData)):
        data[i] = inData[i]+np.random.normal(0, strength, size=(len(data[0]))) # size=(len(data[0]),3) when using raw landmarks
    return data
def negNoise(inData, strength):
    data = inData
    for i in range(len(inData)):
        data[i] = inData[i] + np.random.normal(0, strength, size=(len(data[0]))) # when using raw landmarks
    return data
# </editor-fold>

# <editor-fold desc="Classifiers">
def knnClassifier(data,labels, nNeighbors=20):
    knn = neighbors.KNeighborsClassifier(nNeighbors, weights='distance')
    knn.fit(data,labels)
    return knn

def svmClassifier(data,labels):
    svm = LinearSVC(max_iter= 30000)
    svm = CalibratedClassifierCV(svm)
    svm.fit(data,labels)
    return svm

def randomFores(data, labels, depth):
    rf = RandomForestClassifier(max_depth= 12)
    rf.fit(data,labels)
    return rf

def qda(data, labels):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(data, labels)
    return qda

def lda(data, labels):
    lda = LinearDiscriminantAnalysis()
    lda.fit(data,labels)
    return lda

def dTree(data,labels):
    dt = tree.DecisionTreeClassifier()
    dt.fit(data,labels)
    return dt

def deepNeuralNetwork():
    model = Sequential()
    model.add(Dense(22*22,activation='relu',kernel_initializer='random_normal',input_dim=22*22)) # 48 for raw landmarks 7 for engineered features
    model.add(Dropout(0.20))
    model.add(Dense(150, activation='relu', kernel_initializer='random_normal')) #24 best so far for raw landmarks
    model.add(Dropout(0.20))
    model.add(Dense(150, activation='relu', kernel_initializer='random_normal')) #12 best so far for raw landmarks
    model.add(Dropout(0.20))
    model.add(Dense(150, activation='relu', kernel_initializer='random_normal'))  # 12 best so far for raw landmarks
    model.add(Dropout(0.20))
    model.add(Dense(150, activation='relu', kernel_initializer='random_normal'))  # 12 best so far for raw landmarks
    model.add(Dropout(0.20))
    model.add(Dense(6, activation='softmax', kernel_initializer='random_normal'))
    model.compile(optimizer= adam , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def trainNN(model, data):
    data.features = data.features.reshape(data.features.shape[0], -1)
    data.validData = data.validData.reshape(data.validData.shape[0], -1)
    callback = [EarlyStopping(monitor='val_loss', patience=20), ModelCheckpoint(filepath='weights-best.hdf5', monitor='val_loss', save_best_only=True)]
    fittedNN = model.fit(data.features,data.label,batch_size=batch_size, epochs=epochs,callbacks=callback, verbose = 1,validation_data=(data.validData, data.validLabels))
    fittedNN = load_model('weights-best.hdf5')
    return fittedNN
# </editor-fold>

# Testing
def testClass(classifier, classData, testData, testLabels):
    prediction = classifier.predict(testData)
    accuracy = metrics.accuracy_score(testLabels, prediction)
    print("\n Accuracy: %f" % (accuracy))
    print("\n Results from classifier:  \n %s \n"
          % ( metrics.classification_report(testLabels, prediction)))
    print("\n Confussion matrix:\n %s" % metrics.confusion_matrix(testLabels, prediction))
    cnf_matrix = confusion_matrix(testLabels, prediction)
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=keys, title='Normalized confusion matrix')
    plt.show()

    kfold = 10
    scores = cross_val_score(classifier, classData.features, classData.label, cv=kfold)
    print("\n Cross validation score: \n")
    print(scores)
    print("\n Mean cv score: %f" % (np.mean(scores)))

def predConf(classifier, data):
    features = np.array([data])
    features = calcAllDistanceFeatures(data)
    features = preprocessing.normalize(features)
    features = features.reshape(features.shape[0], -1)
    prob = classifier.predict_proba(features)[0]
    index = np.where(prob == np.amax(prob))[0]
    return index, prob

# Function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'SADNESS', 'SURPRISE')
    y_pos = np.arange(len(objects))  
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()

# Plot the confusion matrix in order to see how our model classified the images
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.RdPu):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


### Main part of driver code to read in do feature engineering, feature selection data augmentation
### Training and finaly classification. Put program in fodler containing a folder with the datafiles with data
### ex ./workDirectory(put program here)/dataSet/000_FEELING.lm3

classifierType = 4 #0 NN, 1 SVM, 2 KNN, 3 LDA/QDA, 4 RF/Desission tree
featureType = 1  # 1 all distances, 2 scaled Engineered features, 3 engineered features
normalisation = 1 # 0 no, 1 yes
dimensionReduction = 0 # 0 none, 1 PCA, 2 Manifold TSNE


dataClass = classData()

if featureType == 900: ### Doesn't work wont convert from list to array
    dataClass.features = dataClass.data # No feature extraction
elif featureType == 1:
    dataClass.features = calcAllDistanceFeatures(dataClass.data) # Using the distance between each feature as the data, 22*22 inputs

elif featureType == 2:
    dataClass.features = scaledFeatEng(dataClass.data) #scaled engineered features from data

elif featureType == 3:
    dataClass.features = featEng(dataClass.data) #engineered features

if normalisation == 1:
    dataClass.features = preprocessing.normalize(dataClass.features)

dataClass.features = dataClass.features.reshape(dataClass.features.shape[0], -1)

if dimensionReduction == 1:
    pca = PCA(n_components=10)
    dataClass.features = pca.fit_transform(dataClass.features)
elif dimensionReduction == 2:
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    dataClass.features = tsne.fit_transform(dataClass.features)


trainData, testData, trainLabels, expectedLabels = holdOut(dataClass, 0.7)

classificationData, classLabel, className = readTestFile()

# <editor-fold desc="Train/Evaluate Classifiers">
if classifierType == 0:
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    epochs = 400
    batch_size = 32

    nnData = classData2()
    nnData.features = trainData
    nnData.label = trainLabels
    trainData, validData, trainLabels, validLabels = holdOut(nnData, 0.7)
    trainData, trainLabels = augment(trainData, trainLabels, 10, 0.01)
    nnData.features = trainData
    nnData.label = trainLabels
    nnData.validData = validData
    nnData.validLabels = validLabels

    nn = deepNeuralNetwork(trainData)
    nn = trainNN(nn, nnData)
    print("Evaluating model...")
    score = nn.evaluate(testData, expectedLabels, verbose=0)
    print('Test accuracy:', score[1])

elif classifierType == 1:
    svm = svmClassifier(trainData, trainLabels)
    print("\n Report for SVM classifier: \n")
    testClass(svm, dataClass, testData, expectedLabels)
    joblib.dump(svm,"svm.dump")

elif classifierType == 2:
    knn = knnClassifier(trainData,trainLabels, 20)
    print("\n Report for K-NN classifier: \n")
    testClass(knn, dataClass, testData, expectedLabels)
    index,probs = predConf(knn,classificationData)
    print(index)
    print(probs)

elif classifierType == 3:
    lda = lda(trainData,trainLabels)
    print("\n Report for LDA classifier: \n")
    testClass(lda, dataClass, testData, expectedLabels)
    qda = qda(trainData,trainLabels)
    print("\n Report for QDA classifier: \n")
    testClass(qda, dataClass, testData, expectedLabels)

elif classifierType == 4:
    rF = randomFores(trainData,trainLabels, 6)
    print("\n Report for Random Forest classifier:")
    testClass(rF, dataClass, testData, expectedLabels)
    dt = dTree(trainData, trainLabels)
    index, probs = predConf(rF,classificationData)
    emotion_ES = list(keys)[index.astype(int)[0]]
    print("\nThe most possible emotion is: {0}".format(emotion_ES))
    print(index)
    print("The possibility for predicated emotions is: {0} ".format(probs))
    emotion_analysis(probs)
    joblib.dump(rF,'randomForest.joblib')
    #print("\n Report for Descision Tree classifier: \n")
    #testClass(dt, dataClass, testData, expectedLabels)
# </editor-fold>
