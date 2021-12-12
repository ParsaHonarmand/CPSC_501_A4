import csv
import numpy as np
import network
from network import saveToFile

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    
def calculateMean(sample):
    L = [float(n) for n in sample if n]
    mean = sum(L)/len(L) if L else '-'
    return mean

def calculateStdDev(sample, mean):
    variance = sum([((float(x) - mean) ** 2) for x in sample]) / len(sample)
    res = variance ** 0.5
    return res

##############################################
# inspired by vampires_net.py code sample
def getDataFromSample(sample):
    mean = 138.3
    stdDev = 20.5
    sbp = cv([standardize(float(sample[1]), mean, stdDev)])

    mean = 3.64
    stdDev = 4.59
    tobacco = cv([standardize(float(sample[2]), mean, stdDev)])

    mean = 4.74
    stdDev = 2.07
    ldl = cv([standardize(float(sample[3]), mean, stdDev)])


    mean = 25.4
    stdDev = 7.77
    adiposity = cv([standardize(float(sample[4]), mean, stdDev)])

    if(sample[5] == "Present"):
        famhist = cv([1])
    elif(sample[5] == "Absent"):
        famhist = cv([0])
    else:
        print("Data processing err. Exiting....")
        quit()

    mean = 53.1
    stdDev = 9.81
    typea = cv([standardize(float(sample[6]), mean, stdDev)])

    mean = 26.0
    stdDev = 4.21
    obesity = cv([standardize(float(sample[7]), mean, stdDev)])

    mean = 53.1
    stdDev = 9.81
    alcohol = cv([standardize(float(sample[8]), mean, stdDev)])

    mean = 42.8
    stdDev = 14.6
    age = cv([standardize(float(sample[9]), mean, stdDev)])

    # concatenate results to get feature vector
    features_vec = np.concatenate((sbp, tobacco, ldl, adiposity, famhist, 
    typea, obesity, alcohol, age), axis=0)
    chd = int(sample[10])
    
    return (features_vec, chd)

# reads number of samples, features, and labels from the file and returns a tuple
# inspired by vampires_net.py sample
def readData(filename):
    with open(filename, newline='') as datafile:
        reader = csv.reader(datafile)        
        # skip the header row
        next(reader, None)  
        
        n = 0
        features = []
        labels = []
        
        for row in reader:
            feature_vec, label = getDataFromSample(row)
            n = n + 1
            features.append(feature_vec)
            labels.append(label)
            
    print("# of data points: " + str(n))
    return n, features, labels


################################################

# reads heart.csv
# returns a tuple for trainingData and testingData, 
# each of which is a zipped array of features and labels
# inspired by vampires_net.py
def prepData():
    n, features, labels = readData('data/heart.csv')

    ntrain = int(n * 5/6)
    ntest = n - ntrain

    train_feature = features[:ntrain]
    train_label = [onehot(label, 2) for label in labels[:ntrain]]
    print("Number of training samples: " + str(ntrain))

    test_feature = features[ntrain:]
    test_label = labels[ntrain:]
    print("Number of testing samples: " + str(ntest))

    trainingData = zip(train_feature, train_label)
    testingData = zip(test_feature, test_label)
    return (trainingData, testingData)


###################################################
trainingData, testingData = prepData()

net = network.Network([9,9,2])
net.SGD(trainingData, 10, 10, .4, test_data = testingData)
saveToFile(net, "part3.pkl")


       