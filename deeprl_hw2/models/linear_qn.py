import tensorflow as tf 
import numpy as np 

import random
import os
from datetime import datetime

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

class LinearQN:
    def __init__(self,model_dir='linear_qn',fixTarget=False,doubleNetwork=False):
        # init some parameters
        self.stepCount = 0
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.stateFrames = STATE_FRAMES
        self.inputH = H
        self.inputW = W
        self.updateTime = UPDATE_TIME
        self.batchSize = BATCH_SIZE
        self.learningRate = LEARNING_RATE
        self.maxIter = NUM_ITERS
        self.memorySize = REPLAY_MEMORY
        self.actionNum = NUM_ACTIONS
        self.hasTarget = fixTarget
        self.model_dir = model_dir
        self.doubleNetwork = doubleNetwork
        self.stateDim = self.inputH*self.inputW*self.stateFrames
        
        # Build model here
        self.W_fc1, self.b_fc1 = self.createNetwork()
        self.W_target, self.b_target = self.createNetwork(isTrainable=doubleNetwork)
        self.resetTarget()
        self.buildModel()
        
        # Start a session and load the model
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        self.session = tf.Session(config=config)
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def createNetwork(self,isTrainable=True):
        # network input
        W_fc1 = tf.Variable(tf.truncated_normal([self.inputH*self.inputW*self.stateFrames,self.actionNum],stddev=0.05),trainable=isTrainable)
        b_fc1 = tf.Variable(tf.zeros([self.actionNum]),trainable=isTrainable)
        return W_fc1, b_fc1
        
    def resetTarget(self):
        if self.hasTarget and not self.doubleNetwork:
            tf.assign(self.W_target, self.W_fc1)
            tf.assign(self.b_target, self.b_fc1)
            print("Target network reset")
            
    # Use target network to forward
    def forwardTarget(self,state_input,isActive_input):
        if self.doubleNetwork:
            (fc_W,fc_b) = tf.cond(self.coinflip[0]>=self.thres[0],lambda: (self.W_fc1,self.b_fc1),lambda: (self.W_target,self.b_target))
        else:
            (fc_W,fc_b) = (self.W_target,self.b_target)
            
        q_vec = tf.matmul(tf.reshape(state_input, [-1, self.stateDim]), fc_W)+fc_b
        q_val = tf.multiply(tf.reduce_max(q_vec,axis=1,keep_dims=True),isActive_input)
        
        if self.doubleNetwork:
            # Do not train the target
            q_val = tf.stop_gradient(q_val)
        return q_val
    
    def forwardWithAction(self,state_input,action_input):
        if self.doubleNetwork:
            (fc_W,fc_b) = tf.cond(self.coinflip[0]<self.thres[0],lambda: (self.W_fc1,self.b_fc1),lambda: (self.W_target,self.b_target))
        else:
            (fc_W,fc_b) = (self.W_fc1,self.b_fc1)
        
        format_input = tf.reshape(state_input, [-1, self.stateDim])
        q_vec = tf.matmul(format_input, fc_W)+fc_b
        actions = tf.one_hot(tf.reshape(action_input, [-1]), self.actionNum, dtype=np.float32) 
        q_val = tf.reduce_sum(tf.multiply(q_vec,actions),axis=1,keep_dims=True)
        return q_val
        
    def forwardWithoutAction(self,state_input,isActive_input):
        q_vec = tf.matmul(tf.reshape(state_input, [-1, self.stateDim]), self.W_fc1)+self.b_fc1
        q_val = tf.multiply(tf.reduce_max(q_vec,axis=1,keep_dims=True),isActive_input)
        return q_val
    
    def getLoss(self):
        # Use huber loss for more robust performance
        self.delta = self.pred_q - self.target_q 
        self.delta = tf.where(tf.abs(self.delta) < 1.0,
                              0.5 * tf.square(self.delta),
                              tf.abs(self.delta) - 0.5, name='clipped_error')

        self.batch_loss = tf.reduce_mean(self.delta, name='loss')
        #self.batch_loss = tf.reduce_mean(tf.square(self.delta), name='loss')
    
    def buildModel(self):
        self.curr_state = tf.placeholder(tf.float32,[self.inputH, self.inputW, self.stateFrames])
        if self.doubleNetwork:
            curr_qvals = tf.matmul(tf.reshape(self.curr_state, [-1, self.stateDim]), self.W_fc1)+self.b_fc1 + \
                         tf.matmul(tf.reshape(self.curr_state, [-1, self.stateDim]), self.W_target)+self.b_target
        else:
            curr_qvals = tf.matmul(tf.reshape(self.curr_state, [-1, self.stateDim]), self.W_fc1)+self.b_fc1
        self.next_action = tf.argmax( curr_qvals, axis=1 )
            
        self.state_input = tf.placeholder(tf.float32,[None, self.inputH, self.inputW, self.stateFrames])
        self.action_input = tf.placeholder(tf.int32,[None, 1])
        self.reward_input = tf.placeholder(tf.float32,[None, 1])
        self.nextState_input = tf.placeholder(tf.float32,[None, self.inputH, self.inputW, self.stateFrames])
        # 1.0 if a terminal state is found
        self.terminal_input = tf.placeholder(tf.float32,[None, 1])
            
        self.isActive_input = tf.ones(tf.shape(self.terminal_input),dtype=tf.float32)-self.terminal_input
        
        self.thres = tf.constant(0.5, shape=[1])
        self.coinflip = tf.random_uniform([1])
        self.pred_q = self.forwardWithAction(self.state_input,self.action_input)
        
        if self.hasTarget:
            self.target_q = self.forwardTarget(self.nextState_input,isActive_input=self.isActive_input)*self.gamma + self.reward_input
        else:
            self.target_q = self.forwardWithoutAction(self.nextState_input,self.isActive_input)*self.gamma + self.reward_input

        self.getLoss()
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # Create the gradient descent optimizer with the given learning rate.
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learningRate)
        self.optimizer = tf.train.RMSPropOptimizer(0.001, momentum=0.95, epsilon=0.01)
        self.grads_and_vars = self.optimizer.compute_gradients(self.batch_loss, tf.trainable_variables())
        self.train_op = self.optimizer.minimize(self.batch_loss, global_step=self.global_step)
        
    def saveModel(self):
        dt = datetime.now().strftime('%m-%d-%H:%M:%S')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, dt)
        self.saver.save(self.session, model_path, global_step=self.global_step)
        print "Model saved to", model_path
