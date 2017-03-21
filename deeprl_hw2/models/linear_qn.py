import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

# Hyper Parameters:
GAMMA = 0.99 # decay rate of past observations
LEARNING_RATE = 0.0001
EPSILON = 0.05
STATE_FRAMES = 4
NUM_ITERS = 5000000
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 10000 # target q-network reset interval
W = 84
H = 84
NUM_ACTIONS = 3
MODEL_DIR = 'linear/no_replay'

class LinearQN:
    def __init__(self,fixTarget=False):
        # init some parameters
        self.stepCount = 0
        self.epsilon = EPSILON
        self.stateFrames = STATE_FRAMES
        self.inputH = H
        self.inputW = W
        self.updateTime = UPDATE_TIME
        self.batchSize = BATCH_SIZE
        self.maxIter = NUM_ITERS
        self.memorySize = REPLAY_MEMORY
        self.actionNum = NUM_ACTIONS
        self.hasTarget = fixTarget
        
        # Build model here
        self.buildModel()
        
        # Start a session and load the model
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
        
        self.resetTarget()
        
    def resetTarget(self):
        if self.hasTarget:
            self.targetW = tf.identity(self.W_fc1)
            self.targetB = tf.identity(self.b_fc1)
            
    # Use target network to forward
    def forwardTarget(self,state_input,isActive_input):
        q_vec = tf.matmul(state_input, self.targetW)+targetB
        q_val = tf.multiply(tf.reduce_max(q_vec,axis=1),isActive_input)
        return q_val
        
    # map state to Q-value vector
    def forward(self,state_input,action_input=None,isActive_input=None):
        # Get q value
        q_vec = tf.matmul(state_input, self.W_fc1)+self.b_fc1
        if isActive_input is not None:
            q_val = tf.multiply(tf.reduce_max(q_vec,axis=1),isActive_input)
            # No backprop
            q_val = tf.stop_gradient(q_val)
        elif action_input is not None:
            actions = tf.one_hot(action_input, self.actionNum, dtype=np.float64) 
            q_val = tf.reduce_sum(tf.multiply(q_vec,actions),axis=1)
        else:
            raise Exception("Either action or terminal must be provided")
            
        return q_val
    
    def getLoss(self, pred_vals, target_vals):
        # Use huber loss for more robust performance
        delta = target_vals - pred_vals 
        clipped_error = tf.where(tf.abs(delta) < 1.0,
                                 0.5 * tf.square(delta),
                                 tf.abs(delta) - 0.5, name='clipped_error')

        loss = tf.reduce_mean(clipped_error, name='loss')
        return loss
    
    def getAction(self):
        self.single_state_input = tf.placeholder(tf.float64,[self.inputH, self.inputW, self.stateFrames])
        q_vec = tf.matmul(self.single_state_input, self.W_fc1)+self.b_fc1
        action = tf.argmax(q_vec, axis=1)
        return action
        
    def buildModel(self):
        self.state_input = tf.placeholder(tf.float64,[None, self.inputH, self.inputW, self.stateFrames])
        self.action_input = tf.placeholder(tf.int32,[None])
        self.reward_input = tf.placeholder(tf.float64,[None])
        self.nextState_input = tf.placeholder(tf.float64,[None, self.inputH, self.inputW, self.stateFrames])
        # 1.0 if a terminal state is found
        self.terminal_input = tf.placeholder(tf.float64,[None])
        
        isActive_input = tf.ones(tf.shape(self.terminal_input),dtype=tf.float64)-self.terminal_input
        
        self.pred_q = self.forward(self.state_input,action_input=self.action_input)
        if self.hasTarget:
            self.target_q = self.forwardTarget(self.nextState_input,terminal_input=isActive_input) + self.reward_input
        else:
            self.target_q = self.forward(self.nextState_input,terminal_input=isActive_input) + self.reward_input
        self.batch_loss = self.getLoss(self.pred_q,self.target_q)
        
        # Create the gradient descent optimizer with the given learning rate.
        self.optimizer = tf.train.GradientDescentOptimizer(self.learningRate)
        self.train_op = self.optimizer.minimize(self.batch_loss)
