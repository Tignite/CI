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


    inputArray = np.array(np.linspace(-10,10,1001))
    wantedOutput = np.array(np.cos(inputArray/2) + np.sin(5/((np.abs(inputArray)+0.2)))-0.1*inputArray)

    def fermi(self, x):
        return 1 / ( 1 + np.exp(-x) )

    def plotGraph(self):
        plt.figure()
        plt.plot(self.inputArray,self.wantedOutput, label ='$cos(x/2) + sin(5/(|x|+0.2))–0.1∙x$')
        plt.plot(self.inputArray,self.mittlererQuadratischerFehler, label ='Mittlerer quadratischer Fehler')
        plt.legend()
        plt.show()
    #Array aller Matrizen
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
            #print(self.activations[layer])
            #Testausgabe
            #print(weightMatrix)
            #print("\n___________________________________________________________\n")


    def forwardPropagate(self, data):
        buffer = [1]
        data = np.array(data)
        if (data.size) > 1:
            for i in range (data.size):
                buffer.append(data[i])
        else:
            #print(data)
            buffer.append(data)
        buffer = np.array(buffer)
        #print(buffer)
        
        for i in range(self.layers + 1):
            if i == 0:
                inputVector = np.array([1])
                self.activations[i] = np.array(np.matmul(self.weights[0], buffer))
            else:
                self.activations[i] = np.array(self.fermi(np.matmul(self.weights[i], np.transpose(self.activations[i-1]))))
            if i < self.layers:
                self.activations[i] = np.insert(self.activations[i], 0, 1, axis = 1)
            #print(self.activations[i])
        return self.activations[len(self.activations) - 1]

    def getCost(self, givenOutput, i):
        #print(i, givenOutput)
        #print(i, givenOutput[0])        
        #print(i, givenOutput[0][0])
        #print(i, self.wantedOutput)
        #print(i, self.wantedOutput[i])
        return np.power(givenOutput[0][0] - self.wantedOutput[i], 2)

    def mittlereKosten(self, i):
        #print("mittlere kosten, errors[] = " , self.errors)
        if i == 0:
            self.mittlererQuadratischerFehler.append(self.errors[i])
        else:
            self.mittlererQuadratischerFehler.append(
                (self.mittlererQuadratischerFehler[i-1] * (i - 1) + self.errors[i]) / i)

    def train(self):
        dataSetSize = 1001
        for i in range (dataSetSize):
            self.trainingOutputs.append(self.forwardPropagate(self.inputArray[i]))
            self.errors.append(self.getCost(self.trainingOutputs[i], i))
            self.mittlereKosten(i)
        print(self.mittlererQuadratischerFehler[1000])
        self.plotGraph()
        
net = ANN()
net.createFullMesh()
net.train()
