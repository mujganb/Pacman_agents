# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
import operator




# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
        
        
        # training the whole data and
        # returning prior & probabilities 
        self.probs, self.prior = self.naive_train(self.data, self.target)
        
        
    
# =============================================================================
#     NAIVE BAYES TRAINING 
# =============================================================================
    
    def naive_train(self, X, y):
    
        # calculating prior probabilities
        prior = np.unique(y,return_counts = True)[1] 
        

        # splitting data based on the labels and
        # storing each label's feature vectors in a dict
        category = {0:[], 1:[], 2:[], 3:[]}
        for idx, i in enumerate(y):
            if i == 0: category[0].append(X[idx])
            if i == 1: category[1].append(X[idx])
            if i == 2: category[2].append(X[idx])
            if i == 3: category[3].append(X[idx])
        
        
        probs = {0:[], 1:[], 2:[], 3:[]}
        for value, content in category.iteritems():
            # calculating the sum of each label
            for i in np.asarray(content).sum(axis=0): # and finding the probability
                probs[value].append((i+1)/(len(category[value])))
            
        return probs, prior # the probability and priors list is returned
    

    # a test function to get a classification when feature vector is passed
    def naive_test(self, test):
    
        rslt = {}
        
        # iterating to find probabilities
        for i in range(len(self.probs)):
        
            probability = []
            probability.append(self.prior[i]/float(sum(self.prior))) # finding prior probability
        
            
            # iterating the new test set to find the probability of each class
            for j in range(len(test)):
                if test[j] == 0: probability.append(1 - self.probs[i][j])
                else: probability.append(self.probs[i][j])
        
            # taking the product of each element in the probability list 
            # considering there is independence between attributes
            probability = np.prod(np.array(probability)) 
            rslt[i] = probability
            
        # return the label with max probability
        # which is the best move to make
        return max(rslt.iteritems() , key=operator.itemgetter(1))[0]
    
   
    
    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
    
        
        # using the naive bayes classifier implemented to classify the vector
        classification = self.naive_test(features)
        
        # converting classification to a move
        move = self.convertNumberToMove(classification)
        
        
        # Get the actions we can try.
        legal = api.legalActions(state)


        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to the API so it can do some safety
        # checking.
        
        # if the converted classifier / move can be performed
        if move in legal: 
            return api.makeMove(move, legal) # then move in that direction
        
        # otherwise move randomly
        # choose a random move from legal
        else: 
            return api.makeMove(random.choice(legal), legal)
    
    
    