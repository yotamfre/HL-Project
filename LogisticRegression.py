import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from matplotlib import pyplot as plt

def ShowData():
    Batch = 4000
    sizeTest = 100
    SizeTrain = Batch - sizeTest

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    trainset = datasets.MNIST('TrainSet', download=True, train=True, transform=transform)
    valset = datasets.MNIST('ValSet', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    #initializing the test and validation sets
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images = images.numpy()

    for i in range(4000):
        for r in range(28):
            for c in range(28):
                if (images[i][0][r][c] > 0):
                    images[i][0][r][c] = 1
                else:
                    images[i][0][r][c] = -1


    images = np.resize(images,(Batch,784))

    X = images[0:SizeTrain,:]
    Y = labels.numpy()[0:SizeTrain]

    new_column = np.ones(SizeTrain)
    X = np.column_stack([X, new_column])

    Xval = images[SizeTrain:Batch,:]
    Yval = labels[SizeTrain:Batch]

    new_column = np.ones(sizeTest)
    Xval = np.column_stack([Xval, new_column])

    #makes the lables batchSize x 10 matrix qith 0's and 1's
    Yvec = np.zeros(shape=(SizeTrain,10))
    for i in range(Y.shape[0]):
        Yvec[i,Y[i]] = 1


    # Training method
    def one_vs_all_logreg(X, y, learning_rate, lamb):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x/100))
        
        # Number of classes
        K = 10

        # Number of features
        N = 785

        # Initialize weights
        weights = np.random.rand(K,N)

        #Training Iterations
        trainIters = 1000

        cost = np.zeros(trainIters)


        # Number of training examples
        m = X.shape[0]
        
        for i in range(trainIters):
            
            # Initialize the cost
            J = 0
            
            # Compute the hypothesis
            h = np.dot(X, weights.T)

            # Compute the cost
            for k in range(K):
                temp = -y[:,k] * np.log(sigmoid(h[:,k])) - (1 - y[:,k]) * np.log(1.000001 - sigmoid(h[:,k]))
                J += np.sum(temp)

                g = np.dot(X. T, (sigmoid(h)[:,k] - y[:,k]))
                g = g/m
                g = g + (lamb / m * np.max(weights[1:,k]))
                

                #update weights 
                weights[k] = weights[k] - learning_rate * g
                
            J += ((lamb / 2) * np.sum(np.square(weights[1:,k])))
            J = J/m
            cost[i] = J



        return (weights,cost)

    def predict(X, weights):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x/100))
        
        return np.argmax(sigmoid(np.dot(X, weights.T)), axis=1)

    theta = one_vs_all_logreg(X, Yvec, 20, .00000001)

    #plots cost
    costList = theta[1]
    plt.plot(costList)
    plt.show()

    weightMat = theta[0]


    #checks the model on the validation set
    predTest = predict(Xval, weightMat)
    numTestFalse = 0
    for i in range(predTest.shape[0]):
        if predTest[i] == Yval[i]:
            numTestFalse = numTestFalse
        else:
            numTestFalse = numTestFalse + 1
    print("Percent incorrect in the validation set" + str(numTestFalse/sizeTest))

    return weightMat

def predict(X, weights):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x/100))
    
    return np.argmax(sigmoid(np.dot(X, weights.T)))

def imageToVecotr(image):
    vec = np.ones(785)
    vec1 = np.resize(image,(1,784))
    vec[1:] = vec1
    return vec