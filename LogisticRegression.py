import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from matplotlib import pyplot as plt

class DigitRecognition():
    global trainloader
    global valloader
    global model
    global lossFunc
    global optimizer

    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

        trainset = datasets.MNIST('TrainSet', download=True, train=True, transform=transform)
        valset = datasets.MNIST('ValSet', download=True, train=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

        self.model = nn.Sequential(nn.Linear(784, 128),
                                 nn.ReLU(),
                                 nn.Linear(128,10),
                                 nn.LogSoftmax(dim=1))

        self.lossFunc = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-3, momentum=0.9)


    def train(self):
        numIters = 5
        lossList = []

        for i in range(numIters):
            runLoss = 0
            for images, labels in self.trainloader:
                images = images.view(images.shape[0], -1)

                out = self.model(images)
                loss = self.lossFunc(out, labels)

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

                runLoss += loss
            lossList.append(runLoss.detach().numpy())
            print("training iteration number " + str(i))

        plt.plot(lossList)
        plt.show()

        #computes accuracy rate on validation set
        dataiter = iter(self.valloader)
        images, labels = dataiter.next()
        images = images.view(images.shape[0], -1)

        out = torch.argmax(self.model(images), 1)

        numCorrect = 0
        for i in range(out.size()[0]):
            if (out[i] == labels[i]):
                numCorrect += 1

        print("percent correct in the validation set is " + str(numCorrect / .64))


    def predict(self, inputs):
        return torch.argmax(self.model(torch.softmax(inputs)), dim=1)
    
    def toVector(image):
        return image.view(image.shape[0], -1)