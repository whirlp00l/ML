import time
import random
import numpy as np

from mnist import MNIST

np.seterr(all = 'ignore')

# transfer functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

# using softmax as output layer is recommended for classification where outputs are mutually exclusive
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

# using tanh over logistic sigmoid for the hidden layer is recommended   
def tanh(x):
    return np.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y*y


class MLP_Classifier(object):
    """
    Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
    Consists of three layers: input, hidden and output.
    """
    def __init__(self, input, hidden, output, iterations = 10, learning_rate = 0.1, 
                output_layer = 'logistic', verbose = True):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        iterations: how many epochs
        learning_rate: initial learning rate
        :param output_layer: activation (transfer) function of the output layer
        :param verbose: whether to spit out error rates while training
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.output_activation = output_layer
        
        # initialize arrays
        self.input = input
        self.hidden = hidden 
        self.output = output

        # set up array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)

        # create randomized weights
        # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
        input_range = 1.0 / self.input ** (1/2)
        self.wi = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.wo = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
        
        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):

        # input activations
        self.ai = inputs

        # hidden activations
        sum = np.dot(self.wi.T, self.ai)
        self.ah = tanh(sum)
        
        # output activations
        sum = np.dot(self.wo.T, self.ah)
        if self.output_activation == 'logistic':
            self.ao = sigmoid(sum)
        elif self.output_activation == 'softmax':
            self.ao = softmax(sum)
 
        return self.ao

    def backPropagate(self, targets):

        # calculate error terms for output
        # the delta (theta) tell you which direction to change the weights
        if self.output_activation == 'logistic':
            output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
        elif self.output_activation == 'softmax':
            output_deltas = -(targets - self.ao)

        # calculate error terms for hidden
        # delta (theta) tells you which direction to change the weights
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = dtanh(self.ah) * error
        
        # update the weights connecting hidden to output, change == partial derivative
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0],1))
        self.wo -= self.learning_rate * change + self.co
        self.co = change 

        # update the weights connecting input to hidden, change == partial derivative
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        self.wi -= self.learning_rate * change + self.ci
        self.ci = change

        # calculate error
        if self.output_activation == 'softmax':
            error = -sum(targets * np.log(self.ao))
        elif self.output_activation == 'logistic':
            error = sum(0.5 * (targets - self.ao)**2)
        
        return error

    def fit(self, inputs, targets):
        if self.verbose == True:
            if self.output_activation == 'softmax':
                print 'Using softmax activation in output layer'
            elif self.output_activation == 'logistic':
                print 'Using logistic sigmoid activation in output layer'
                
        num_example = len(inputs)
                
        for i in range(self.iterations):
            error = 0.0
            for i in range(num_example):
                input = inputs[i]
                target = targets[i]
                self.feedForward(input)
                error += self.backPropagate(target)
                
            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()
            
            error = error/60000
            print('Training error %-.5f' % error)
                
    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            tmp = np.argmax(self.feedForward(p))
            predictions.append(tmp)
            #print tmp
        return predictions

    def load_data(self):
        mn = MNIST('.')
        mn.test()
        data = mn.train_images
        data = np.array(data)
        
        data.astype(np.float32)
        data = data/255.0
        return data

    def load_targets(self):
        mn = MNIST('.')
        mn.test()
        targets = []
        for t in mn.train_labels:
            #print t
            out = np.zeros(self.output)
            out[t] = 1
            targets.append(out)
        targets = np.array(targets)
        return targets




if __name__ == '__main__':
    mn = MNIST('.')
    mn.test()
    MLP = MLP_Classifier(28*28+1,40,10)
    datas = MLP.load_data()
    targets = MLP.load_targets()
    MLP.fit(datas,targets)
    result = MLP.predict(mn.test_images)
    accurate = 0
    for i in range(len(result)):
        if result[i] == mn.test_labels[i]:
            accurate+=1
    print accurate