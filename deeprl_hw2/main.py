from atari_environment import AtariEnv
from replay_memory import Sample, ReplayMemory
from models.linear_qn import LinearQN

import time
import numpy as np
import random

# Training parameters
epsilon_init = 1.0
epsilon_final = 0.01
epsilon_decay_steps = 1000000
epsilon_step = (epsilon_final-epsilon_init)/epsilon_decay_steps

batch_size = 32

# Training periods
n_train = 5000000
replay_size = 1000000
initial_buffer = 50000
target_reset_freq = 10000
model_save_freq = 100000

# Create environent and model
env_name = 'SpaceInvaders-v0'
do_render = True
fix_target = True
env = AtariEnv(env_name, do_render=do_render)
model = LinearQN(fixTarget=fix_target)

sample_from_replay = True # False for Q2
if sample_from_replay:
    D = ReplayMemory(replay_size)

def train():
    sess = model.session
    train_counter = 0
    ep = 0
    epsilon = epsilon_init
    while train_counter<n_train:
        # Within an episode
        episode_local_counter = 0
        state, _, _ = env.new_game()
        
        step_time = 0.
        total_loss = 0.
        accum_reward = 0
        
        while True:
            _tt = time.time()
            # Add to memory only
            if sample_from_replay and len(D) <= initial_buffer:
                action = env.random_action()
                next_state, reward, is_terminal = env.step(action)
                if is_terminal:
                    break
                D.append(state, action, reward, next_state, is_terminal)
                continue
            
            epsilon += epsilon_step
            
            if random.random() < epsilon: # uniform_random
                action = env.random_action()
            else: # get action from qn
                _t = time.time()
                action = sess.run(model.next_action, {
                    model.curr_state: state / 255.
                })[0]
                # print 'model.next_action', time.time() - _t
            next_state, reward, is_terminal = env.step(action)
            
            if is_terminal:
                break
            
            train_counter += 1
            episode_local_counter += 1
            accum_reward += reward
            
            if sample_from_replay: # sample minibatch from D
                D.append(state, action, reward, next_state, is_terminal)
                samples = D.sample(batch_size)
                loss = _train_on_samples(model, samples)
            else: # on-policy
                samples = [Sample(state, action, reward, next_state, is_terminal)]
                loss = _train_on_samples(model, samples)
            # print 'each step', time.time() - _tt

            step_time += time.time()-_tt
            total_loss += loss

            if train_counter % target_reset_freq == 0:
                model.resetTarget()
            
            if train_counter % model_save_freq == 0:
                model.saveModel()
        
        if episode_local_counter == 0:
            print 'current buffer size: %d' % len(D)
        else:
            ep += 1
            print 'episode {0}:\t {1} steps, accum_reward: {2}, loss: {3}'.format(ep, episode_local_counter, accum_reward, total_loss/episode_local_counter)
            print '===== average step_time: %f'%(step_time/episode_local_counter)
            print '===== total iter: {0}'.format(train_counter)

def _train_on_samples(model, samples):
    _t = time.time()
    sess = model.session
    state_list = np.array([s.state for s in samples]).astype(np.float32) / 255.
    action_list = np.array([[s.action] for s in samples])
    reward_list = np.array([[s.reward] for s in samples])
    next_state_list = np.array([s.next_state for s in samples]).astype(np.float32) / 255.
    is_terminal_list = np.array([[s.is_terminal] for s in samples]) + 0. # True -> 1
    
    _, loss = sess.run([model.train_op, model.batch_loss], {
        model.state_input: state_list,
        model.action_input: action_list,
        model.reward_input: reward_list,
        model.nextState_input: next_state_list,
        model.terminal_input: is_terminal_list
    })
    # print '_train_on_samples', time.time() - _t
    return loss

train()
