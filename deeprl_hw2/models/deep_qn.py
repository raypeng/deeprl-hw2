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
MODEL_PATH = "dqn/dqn"
MODEL_DIR = "dqn"

class DQNetwork:
    def __init__(self,components):
        [W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2] = components

        self.W_conv1 = W_conv1
        self.b_conv1 = b_conv1
        self.W_conv2 = W_conv2
        self.b_conv2 = b_conv2
        self.W_conv3 = W_conv3
        self.b_conv3 = b_conv3
        self.W_fc1 = W_fc1
        self.b_fc1 = b_fc1
        self.W_fc2 = W_fc2
        self.b_fc2 = b_fc2
    
    def copy(self, m):
        tf.assign(self.W_conv1, m.W_conv1)
        tf.assign(self.b_conv1, m.b_conv1)
        tf.assign(self.W_conv2, m.W_conv2)
        tf.assign(self.b_conv2, m.b_conv2)
        tf.assign(self.W_conv3, m.W_conv3)
        tf.assign(self.b_conv3, m.b_conv3)
        tf.assign(self.W_fc1, m.W_fc1)
        tf.assign(self.b_fc1, m.b_fc1)
        tf.assign(self.W_fc2, m.W_fc2)
        tf.assign(self.b_fc2, m.b_fc2)

class DeepQN:
    def __init__(self,doubleNetwork=False):
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
        self.doubleNetwork = doubleNetwork
        self.stateDim = self.inputH*self.inputW*self.stateFrames
        
        # Build model here
        self.model_active = DQNetwork(self.createNetwork())
        self.model_target = DQNetwork(self.createNetwork(isTrainable=False))
        
        self.resetTarget()
        self.buildModel()
        
        # Start a session and load the model
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    
        
    def createNetwork(self,isTrainable=True):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32],isTrainable)
        b_conv1 = self.bias_variable([32],isTrainable)

        W_conv2 = self.weight_variable([4,4,32,64],isTrainable)
        b_conv2 = self.bias_variable([64],isTrainable)

        W_conv3 = self.weight_variable([3,3,64,64],isTrainable)
        b_conv3 = self.bias_variable([64],isTrainable)

        W_fc1 = self.weight_variable([3136,512],isTrainable)
        b_fc1 = self.bias_variable([512],isTrainable)

        W_fc2 = self.weight_variable([512,self.actionNum],isTrainable)
        b_fc2 = self.bias_variable([self.actionNum],isTrainable)

        return [W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2]
    
    def networkForward(self,model,input):
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(input,model.W_conv1,4) + model.b_conv1)
        #h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,model.W_conv2,2) + model.b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,model.W_conv3,1) + model.b_conv3)
        
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,model.W_fc1) + model.b_fc1)
        q_vec = tf.matmul(h_fc1,model.W_fc2) + model.b_fc2
        return q_vec
        
    def resetTarget(self):
        self.model_target.copy(self.model_active)
        print("Target network reset")
            
    # Use target network to forward
    def forwardTarget(self,state_input,isActive_input):
        if self.doubleNetwork:
            q_vec = self.networkForward(self.model_active, state_input)
        else:
            q_vec = self.networkForward(self.model_target, state_input)

        q_val = tf.multiply(tf.reduce_max(q_vec,axis=1,keep_dims=True),isActive_input)
        return q_val
    
    def forwardWithAction(self,model,state_input,action_input):
        q_vec = self.networkForward(model, state_input)
        actions = tf.one_hot(tf.reshape(action_input, [-1]), self.actionNum, dtype=np.float32) 
        q_val = tf.reduce_sum(tf.multiply(q_vec,actions),axis=1,keep_dims=True)
        return q_val
        
    def getLoss(self):
        # Use huber loss for more robust performance
        self.delta = self.pred_q - self.target_q 
        self.delta = tf.where(tf.abs(self.delta) < 1.0,
                              0.5 * tf.square(self.delta),
                              tf.abs(self.delta) - 0.5, name='clipped_error')

        self.batch_loss = tf.reduce_mean(self.delta, name='loss')
    
    def buildModel(self):
        self.curr_state = tf.placeholder(tf.float32,[self.inputH, self.inputW, self.stateFrames])
        curr_state_reshape = tf.reshape(self.curr_state, [1, self.inputH, self.inputW, self.stateFrames])
        
        self.next_action = tf.argmax( self.networkForward(self.model_active, curr_state_reshape), axis=1 )
            
        self.state_input = tf.placeholder(tf.float32,[None, self.inputH, self.inputW, self.stateFrames])
        self.action_input = tf.placeholder(tf.int32,[None, 1])
        self.reward_input = tf.placeholder(tf.float32,[None, 1])
        self.nextState_input = tf.placeholder(tf.float32,[None, self.inputH, self.inputW, self.stateFrames])
        # 1.0 if a terminal state is found
        self.terminal_input = tf.placeholder(tf.float32,[None, 1])
            
        self.isActive_input = tf.ones(tf.shape(self.terminal_input),dtype=tf.float32)-self.terminal_input
            
        self.pred_q = self.forwardWithAction(self.model_active,self.state_input,self.action_input)
        
        self.target_q = self.forwardTarget(self.nextState_input,isActive_input=self.isActive_input)*self.gamma + self.reward_input

        self.getLoss()
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # Create the gradient descent optimizer with the given learning rate.
        self.optimizer = tf.train.RMSPropOptimizer(0.00025,decay=0.95, momentum=0.95, epsilon=0.01)
        self.train_op = self.optimizer.minimize(self.batch_loss, global_step=self.global_step)
        
    def saveModel(self):
        self.saver.save(self.session, MODEL_PATH, global_step=self.global_step)
    
    def weight_variable(self,shape,isTrainable=True):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial, trainable=isTrainable)

    def bias_variable(self,shape,isTrainable=True):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial, trainable=isTrainable)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")