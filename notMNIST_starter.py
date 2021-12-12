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

def standardize(x, mu, sigma):
    return ((x-mu)/sigma)
    
#################################################################
def getImgData(feature_vectors):
    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    features = []
    for image in feature_vectors:
        reshaped_image = np.array(image).reshape((784, 1)) / 255
        features.append(reshaped_image)
    features = np.array(features, dtype=np.float128)
    return features

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    final_training_labels = [onehot(label, 10) for label in train_labels]

    final_train_features = getImgData(train_features)
    final_test_features = getImgData(test_features)

    trainingData = zip(final_train_features, final_training_labels)
    testingData = zip(final_test_features, test_labels)
    return (trainingData, testingData)
    
###################################################################


trainingData, testingData = prepData()

net = network.Network([784,40,10])
net.SGD(trainingData, 12, 10, 1.0, test_data = testingData)
saveToFile(net, "part2.pkl")








