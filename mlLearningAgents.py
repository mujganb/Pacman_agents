# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import math

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        
        
        # state action pairs of Q values
        self.qValues = util.Counter() # dict with default 0
        
        # set score to zero
        self.score = 0
        # and stored all actions and states in a list
        self.lastState = []
        self.lastAction = []
        
# =============================================================================
# =============================================================================

    def getQ_Value(self, state, action): # returns Q-value in specified state action pair
        return self.qValues[(state,action)]
    
    def getQ_Max(self, state): # returns maximum Q-value
        q_val = []
        for action in state.getLegalPacmanActions(): 
            q = self.getQ_Value(state, action) # gets Q-value of each action
            q_val.append(q) # appends to list
            
        if len(q_val) == 0: return 0.0 # if none action is found, return 0
        else: return max(q_val) # returns max value
    
    def updateQ_Value(self, state, action, q_max, reward):
        if action != 'Stop':
            
            q = self.getQ_Value(state, action) 
            # updates Q-value by rule
            self.qValues[(state,action)] = q + self.alpha*(reward+self.gamma*q_max - q) 
    
    def ghost_remove(self, legal, state):
        
        position_pacman = state.getPacmanState().configuration.pos 
        position_ghost = state.getGhostPositions()
        dist0 = int(position_ghost[0][0]) - int(position_pacman[0]) # find the distance of ghost from pacman in x axis
        dist1 = int(position_ghost[0][1]) - int(position_pacman[1]) # finds the distance in y axis
        

        # remove action if ghost is on that direction
        # if distance is 1 in x direction, ghost is on the right -> east
        # if distance is 2, then ghost is on that direction, so no need to move to that direction
        if (dist0 == 1 or dist0 == 2) and dist1 == 0: 
            if Directions.EAST in legal and len(legal) != 1:
                legal.remove(Directions.EAST)
        # same applies to each direction
        elif (dist0 == -1 or dist0 == -2) and dist1 == 0:
            if Directions.WEST in legal and len(legal) != 1:
                legal.remove(Directions.WEST)
        elif dist0 == 0 and (dist1 == 1 or dist1 == 2):
            if Directions.NORTH in legal and len(legal) != 1:
                legal.remove(Directions.NORTH)
        elif dist0 == 0 and (dist1 == -1 or dist1 == -2):
            if Directions.SOUTH in legal and len(legal) != 1:
                legal.remove(Directions.SOUTH)        

        # also to prevent ghost to turn back  while not being chased by ghost
        if len(self.lastAction) > 0:
            last_action = self.lastAction[-1]
            # if the euclidean distance is greater than 2 means ghost is not near
            if math.sqrt(abs(dist0)**2 + abs(dist1)**2) > 2:
                # remove the last action if its in legal actions
                # so that pacman won't return back for efficiency
                if (Directions.REVERSE[last_action] in legal) and len(legal)>1:
                    legal.remove(Directions.REVERSE[last_action])        

        return legal
    
    def choose_action(self, q_val):
        # return highest q-value action's index
        idx = random.randint(0, (len(q_val) - 1)) # select random number between indexes of q_val list
        max_val = q_val[idx] # initialise max value as the value of random selected index
        # search through the list if there is any value greater than selected
        for i in range(len(q_val)):
            if q_val[i] > max_val:
                idx = i
                max_val = q_val[i]
        return idx # return the index
    
# =============================================================================
# =============================================================================
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        
        # positions and score is assigned
        position_pacman = state.getPacmanState().configuration.pos #state.getPacmanPosition()
        position_ghost = state.getGhostPositions()
        current_score = state.getScore()
        
        
        # directions are removed from legal if the ghost is there
        legal = self.ghost_remove(legal, state)
        
        
        print "Legal moves: ", legal
        print "Pacman position: ", position_pacman
        print "Ghost positions:" , position_ghost
        print "Food locations: "
        print state.getFood()
        print "Score: ", current_score
        
        
        # update Q Value
        # first assign reward
        reward = current_score - self.score
        if len(self.lastState) > 0: # then if there are actions before start
            # assign last state and last action
            last_state = self.lastState[-1]
            last_action = self.lastAction[-1]
            # update Q-value by using function created
            self.updateQ_Value(last_state, last_action, self.getQ_Max(state), reward)
        
        
        # Now pick what action to take. For now a random choice among
        # the legal moves
        
        
        # store all the Q-values of each action in a list
        q_val = []
        for i in range(len(legal)):
            q_val.append(self.getQ_Value(position_pacman, legal[i]))
            
        # make epsilon greedy choice
        choice = random.random()
        if choice <= (1 - self.epsilon):
            idx = self.choose_action(q_val) # get max Q-value's index
            pick = legal[idx] # return the action

        else:
            pick = random.choice(legal) 


        # update score, state list and action list
        self.score = state.getScore()
        self.lastState.append(state)
        self.lastAction.append(pick)
        
        
        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        # update Q-values again for training
        reward = state.getScore() - self.score
        self.updateQ_Value(self.lastState[-1], self.lastAction[-1], 0, reward)

        # set score and lists to 0
        self.score = 0
        self.lastState = []
        self.lastAction = []

        print "A game just ended!"
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


