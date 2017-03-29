import tensorflow as tf 
import numpy as np 

import random
import os
from datetime import datetime

# Hyper Parameters:
GAMMA = 0.99 # decay rate of past observations
NUM_ACTIONS = 3

class DQNetwork:
    def __init__(self,components,name):
        [W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2] = components
        self.name = name
        self.W_conv1 = W_conv1
        self.b_conv1 = b_conv1
        self.W_conv2 = W_conv2
        self.b_conv2 = b_conv2
        self.W_fc1 = W_fc1
        self.b_fc1 = b_fc1
        self.W_fc2 = W_fc2
        self.b_fc2 = b_fc2
    
    def copy(self, m):
        tf.assign(self.W_conv1, m.W_conv1)
        tf.assign(self.b_conv1, m.b_conv1)
        tf.assign(self.W_conv2, m.W_conv2)
        tf.assign(self.b_conv2, m.b_conv2)
        tf.assign(self.W_fc1, m.W_fc1)
        tf.assign(self.b_fc1, m.b_fc1)
        tf.assign(self.W_fc2, m.W_fc2)
        tf.assign(self.b_fc2, m.b_fc2)
    
    # Take a list of states, return the greedy action
    def getPolicy(self, s_input):
        q_vec = self.getQVector(s_input)
        actions = tf.argmax( q_vec, axis=1 ,name='get_action')
        return actions
    
    def getMaxQValue(self, s_input, isActive_input):
        q_vec = self.getQVector(s_input)
        q_max = tf.multiply( tf.reduce_max(q_vec,axis=1,keep_dims=True),isActive_input )
        actions = tf.argmax(q_vec, axis=1)
        return q_max, actions
        
    # Take a list of states and actions, return corresponding q values
    def getQValue(self, s_input, a_input):
        q_vec = self.getQVector(s_input)
        actions = tf.one_hot(tf.reshape(a_input, [-1]), NUM_ACTIONS, dtype=np.float32, name='action_onehot') 
        q_val = tf.reduce_sum(tf.multiply(q_vec,actions),axis=1,keep_dims=True, name='get_q_val')
        return q_val
        
    def getQVector(self, s_input):
        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(s_input,self.W_conv1,4) + self.b_conv1, name='conv1')
        h_conv2 = tf.nn.relu(conv2d(h_conv1,self.W_conv2,2) + self.b_conv2, name='conv2')
        # Dimension: 9x9x32
        h_conv2 = tf.reshape(h_conv2, [-1,2592], name='conv2_flat')
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2,self.W_fc1) + self.b_fc1, name='fc1')
        q_vec = tf.matmul(h_fc1,self.W_fc2) + self.b_fc2
        return q_vec
    
class DeepQN:
    def __init__(self,model_dir='dqn',doubleNetwork=False,lr=0.00025,initStd=0.02):
        # init some parameters
        self.gamma = GAMMA
        self.model_dir = model_dir
        self.doubleNetwork = doubleNetwork
        self.lr = lr
        self.initStd = initStd
        
        # Build model here
        self.model_active = DQNetwork(self.createNetwork('active'),'active')
        self.model_target = DQNetwork(self.createNetwork('target',isTrainable=False),'target')
        
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
    
        
    def createNetwork(self,model_name,isTrainable=True):
        with tf.variable_scope(model_name) as scope:
            # network weights
            W_conv1 = self.weight_variable([8,8,4,16],'W_conv1',isTrainable)
            b_conv1 = self.bias_variable([16],'b_conv1',isTrainable)

            W_conv2 = self.weight_variable([4,4,16,32],'W_conv2',isTrainable)
            b_conv2 = self.bias_variable([32],'b_conv2',isTrainable)
            
            W_fc1 = self.weight_variable([2592,256],'W_fc1',isTrainable)
            b_fc1 = self.bias_variable([256],'b_fc1',isTrainable)

            W_fc2 = self.weight_variable([256,NUM_ACTIONS],'W_fc2',isTrainable)
            b_fc2 = self.bias_variable([NUM_ACTIONS],'b_fc2',isTrainable)

        return [W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2]
    
    def resetTarget(self):
        self.model_target.copy(self.model_active)
        print("Target network reset")
    
    def getLoss(self):
        # Use huber loss for more robust performance
        self.delta = self.target_q - self.pred_q 
        self.delta = tf.where(tf.abs(self.delta) < 1.0,
                              0.5 * tf.square(self.delta),
                              tf.abs(self.delta) - 0.5, name='clipped_error')

        self.batch_loss = tf.reduce_mean(self.delta, name='huber_loss')
    
    def buildModel(self):
        # For policy extraction
        self.curr_state = tf.placeholder(tf.float32,[84, 84, 4])
        curr_state_reshape = tf.reshape(self.curr_state, [1, 84, 84, 4])
        self.next_action = self.model_active.getPolicy(curr_state_reshape)
        
        # For training
        self.state_input = tf.placeholder(tf.float32,[None, 84, 84, 4])
        self.action_input = tf.placeholder(tf.int32,[None, 1])
        self.reward_input = tf.placeholder(tf.float32,[None, 1])
        self.nextState_input = tf.placeholder(tf.float32,[None, 84, 84, 4])
        # 1.0 if a terminal state is found
        self.terminal_input = tf.placeholder(tf.float32,[None, 1])
        self.isActive_input = tf.ones_like(self.terminal_input,dtype=tf.float32)-self.terminal_input
        
        self.pred_q = self.model_active.getQValue(self.state_input,self.action_input)
        if self.doubleNetwork:
            _, next_batch_action = self.model_active.getMaxQValue(self.nextState_input, self.isActive_input)
            self.target_q = self.model_target.getQValue(self.nextState_input, next_batch_action)*self.gamma+self.reward_input
        else:    
            self.target_q = self.model_target.getMaxQValue(self.nextState_input, self.isActive_input)
            self.target_q = self.target_q*self.gamma + self.reward_input
            
        self.getLoss()
        
        # Gradient Descent
        #self.summary_writer = tf.train.SummaryWriter('/tensorboard', graph_def=sess.graph_def)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        self.grads_and_vars = self.optimizer.compute_gradients(self.batch_loss, tf.trainable_variables())
        self.train_op = self.optimizer.minimize(self.batch_loss, global_step=self.global_step)
        
    def saveModel(self):
        dt = datetime.now().strftime('%m-%d-%H:%M:%S')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, dt)
        self.saver.save(self.session, model_path, global_step=self.global_step)
        print("Model saved to", model_path)
    
    def weight_variable(self,shape,name,isTrainable=True):
        #initial = tf.truncated_normal(shape, stddev=self.initStd)
        #return tf.Variable(initial, name=name, trainable=isTrainable)
        W = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=isTrainable)
        return W
        
    def bias_variable(self,shape,name,isTrainable=True):
        initial = tf.zeros(shape = shape)
        return tf.get_variable(name, initializer=initial, trainable=isTrainable)

def conv2d(x, W, stride, name='conv2D'):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID", name=name)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
