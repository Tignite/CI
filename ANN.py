import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class ANN:
    def __init__(self):
        self.activations = []
        self.weights = []
        self.errors = []
        self.mittlererQuadratischerFehler = []
        self.trainingOutputs = []
    
    layers = 1
    units = 10
    inputs = 1
    outputs = 1
    learnRate = 0.05
    inputCount = 1001


    inputArray = np.array(np.linspace(-10,10,1001))
    wantedOutput = np.array(np.cos(inputArray/2) + np.sin(5/((np.abs(inputArray)+0.2)))-0.1*inputArray)

    def updateLearnRate(self):
        self.learnRate -= self.learnRateDelta
        
    def fermi(self, x, deriv):
        if (deriv == False):
            return 1 / ( 1 + np.exp(-x) )
        return ( 1 / ( 1 + np.exp(-x) ) ) * ( 1 - ( 1 / ( 1 + np.exp(-x) ) ) )

    def plotGraph(self):
        plt.figure()
        plt.plot(self.inputArray,self.wantedOutput, label ='$cos(x/2) + sin(5/(|x|+0.2))–0.1∙x$')
        plt.plot(self.inputArray,self.mittlererQuadratischerFehler, label ='Mittlerer quadratischer Fehler')
        plt.plot(self.inputArray,self.trainingOutputs, label ='Netzausgabe')
        plt.legend()
        plt.show()
        
    def createFullMesh(self):

        for layer in range (self.layers + 1):
            weightMatrix = []
            innerLimit = 0
            outerLimit = self.units
            if layer == 0:
                innerLimit = self.inputs + 1
            else:
                innerLimit = self.units + 1
                if layer == self.layers:
                    outerLimit = 1
            for unit in range (outerLimit):
                weightVector = []
                for i in range (innerLimit):
                    weightVector.append(rnd.uniform(-0.5,0.5))
                weightMatrix.append(weightVector)
            self.weights.append(np.mat(weightMatrix))
            activationVector = np.zeros((innerLimit - 1 ,outerLimit))
            self.activations.append(np.insert(activationVector, 0, 1, axis = 0))
            
    def forwardPropagate(self, data):
        buffer = [1]
        data = np.array(data)
        if (data.size) > 1:
            for i in range (data.size):
                buffer.append(data[i])
        else:
            buffer.append(data)
        buffer = np.array(buffer)
        
        for i in range(self.layers + 1):
            if i == 0:
                inputVector = np.array([1])
                self.activations[i] = np.array(np.dot(self.weights[0], buffer))
            elif i < self.layers:
                self.activations[i] = np.array(self.fermi(np.dot(self.weights[i], np.transpose(self.activations[i-1])), deriv=False))
            else:
                self.activations[i] = np.array(np.dot(self.weights[i], np.transpose(self.activations[i-1])))
            if i < self.layers:
                self.activations[i] = np.insert(self.activations[i], 0, 1, axis = 1)
        return self.activations[len(self.activations) - 1]
    
    def backPropagate(self, i):
        layer2_error = self.wantedOutput[i] - self.trainingOutputs[i]
        layer2_delta = layer2_error * self.fermi(self.activations[1], deriv=True)
        layer1_error = layer2_delta.T.dot(self.weights[1])
        layer1_delta = layer1_error * self.fermi(self.activations[0], deriv=True).T
        
        work = []
        for j in range (1, self.units + 1):
            work.append(self.activations[0][0][j])
        work = np.mat(work)

        self.weights[1] += self.activations[1].T.dot(layer2_delta) * self.learnRate
        self.weights[0] += work.T.dot(layer1_delta) * self.learnRate
                
    def getCost(self, givenOutput, i):
        return np.power(givenOutput - self.wantedOutput[i], 2)
        
    def mittlereKosten(self, i):
        if i == 0:
            self.mittlererQuadratischerFehler.append(self.errors[i])
        else:
            self.mittlererQuadratischerFehler.append(
                (self.mittlererQuadratischerFehler[i-1] * (i - 1) + self.errors[i]) / i)
            
    def train(self):
        dataSetSize = 1001
        for i in range (dataSetSize):
            out = self.forwardPropagate(self.inputArray[i])[0][0]
            self.trainingOutputs.append(out)
            self.errors.append(self.getCost(self.trainingOutputs[i], i))
            self.mittlereKosten(i)
            self.backPropagate(i)
        print(self.mittlererQuadratischerFehler[1000])
        self.trainingOutputs = np.array(self.trainingOutputs)
        self.plotGraph()
        
net = ANN()
net.createFullMesh()
net.train()
