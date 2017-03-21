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
    def forwardTarget(self,state_input,terminal_input):
        q_vec = tf.matmul(state_input, self.targetW)+targetB
        q_val = tf.multiply(tf.reduce_max(q_vec,axis=1),terminal_input)
        return q_val
        
    # map state to Q-value vector
    def forward(self,state_input,action_input=None,terminal_input=None):
        # network input
        with tf.name_scope('fc1'):
            self.W_fc1 = tf.Variable(tf.truncated_normal([self.inputH,self.inputW,self.stateFrames,self.actionNum]),
                                stddev=0.01,
                                name='weights')
            self.b_fc1 = tf.Variable(tf.zeros([self.actionNum]),
                                name='biases')
            q_vec = tf.matmul(state_input, self.W_fc1)+self.b_fc1
        # Get q value
        if terminal_input is not None:
            q_val = tf.multiply(tf.reduce_max(q_vec,axis=1),terminal_input)
            # No backprop
            q_val = tf.stop_gradient(q_val)
        elif action_input is not None:
            actions = tf.one_hot(action_input, self.actionNum, dtype=np.float64) 
            q_val = tf.reduce_sum(tf.multiply(q_vec,actions),axis=1)
        else:
            raise Exception("Either action or terminal must be provided")
            
        return q_val
    
    def loss(self, pred_vals, target_vals):
        # Use huber loss for more robust performance
        delta = target_vals - pred_vals 
        clipped_error = tf.where(tf.abs(delta) < 1.0,
                                 0.5 * tf.square(delta),
                                 tf.abs(delta) - 0.5, name='clipped_error')

        loss = tf.reduce_mean(clipped_error, name='loss')
        return loss
    
    def buildModel(self):
        self.state_input = tf.placeholder(tf.float64,[None, self.inputH, self.inputW, self.stateFrames])
        self.action_input = tf.placeholder(tf.int32,[None])
        self.reward_input = tf.placeholder(tf.float64,[None])
        self.nextState_input = tf.placeholder(tf.float64,[None, self.inputH, self.inputW, self.stateFrames])
        # 0 if the state is a terminal state
        self.terminal_input = tf.placeholder(tf.float64,[None])
        
        self.pred_q = self.forward(self.state_input,action_input=self.action_input)
        if self.hasTarget:
            self.target_q = self.forwardTarget(self.nextState_input,terminal_input=self.terminal_input) + self.reward_input
        else:
            self.target_q = self.forward(self.nextState_input,terminal_input=self.terminal_input) + self.reward_input
        self.batch_loss = self.loss(self.pred_q,self.target_q)
    