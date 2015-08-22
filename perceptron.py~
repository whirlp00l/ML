import os
import sys
from mnist import MNIST


class Perceptron:
  def __init__( self, classnum, max_iterations):
    self.classnum = classnum
    self.max_iterations = max_iterations
    self.weights = {}
    self.learningrate = 1
    for label in range(classnum):
      self.weights[label] = {}

  def classify(self, data ):
    guesses = []
    for datum in data:
      vectors = {}
      maxScore = -sys.maxint
      maxl = None
      for l in range(self.classnum):
        vectors[l] = 0
        for j in range(len(self.weights[0])):
          vectors[l] += datum[j] * self.weights[l][j]
        maxScore = maxScore if maxScore>vectors[l] else vectors[l]
        maxl = maxl if maxScore>vectors[l] else l
      guesses.append(maxl)
    return guesses  

  def train( self, trainingData, trainingClass, validationData, validationClass ):
    weights = []
    for i in range(self.classnum):
      weights.append([0]*len(trainingData[0]))
    self.weights = weights
    prev_accurate = 0
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..."
      for i in range(len(trainingData)):            
        bestScore = None
        bestClass = None
        datum = trainingData[i]
        for c in range(self.classnum):
          score = 0
          for j in range(len(trainingData[i])):
            score += datum[j] * self.weights[c][j]
          if score > bestScore or bestScore is None:
            bestScore = score
            bestClass = c
      
        actualClass = trainingClass[i]
        if bestClass != actualClass:
          # Wrong guess, update weights
          for k in range(len(trainingData[i])):
            self.weights[actualClass][k] = self.weights[actualClass][k] + datum[k]
            self.weights[bestClass][k] = self.weights[bestClass][k] - datum[k]
      print "Checking result ", iteration, "..."
      guessClass = self.classify(validationData)
      count = 0
      for i in range(len(guessClass)):
        if guessClass[i] == validationClass[i]:
          count=count+1
      accurate = float(count)/float(len(validationClass))
      print "Current accurate", accurate 
      if accurate - prev_accurate < 0.01:
        break;
      else:
        prev_accurate = accurate
    print "Test finished:"
    print "W is "
    for i in range(self.classnum):
      print i," : ",weights[i]
    print "Last Test accurate is: ", accurate

  


#for MNIST data
mn = MNIST('.')
mn.test()
p = Perceptron(10,100)
p.train(mn.train_images,mn.train_labels,mn.test_images,mn.test_labels)
