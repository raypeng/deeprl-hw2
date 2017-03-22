from atari_environment import AtariEnv
from replay_memory import Sample, ReplayMemory
from models.linear_qn import LinearQN

import time
import numpy as np
import random


gamma = 0.99
lr = 1e-4
epsilon = 0.05
n_train = 5000000
replay_size = 1000000
target_reset_freq = 10000
batch_size = 32
M = 200

env_name = 'SpaceInvaders-v0'
env = AtariEnv(env_name)
model = LinearQN()

sample_from_replay = True # False for Q2
if sample_from_replay:
    D = ReplayMemory(replay_size)


def train():
    sess = model.session
    train_counter = 0
    for ep in range(M):
        episode_local_counter = 0
        accum_reward = 0
        state, _, _ = env.new_game()
        while True:
            _tt = time.time()
            if random.random() < epsilon: # uniform_random
                action = env.random_action()
            else: # get action from qn
                _t = time.time()
                action_tensor = model.getAction()
                print 'getAction', time.time() - _t
                action = sess.run(action_tensor, {
                    model.single_state_input: state / 255.
                })[0]
                print 'run action_tensor', time.time() - _t
            next_state, reward, is_terminal = env.step(action)
            if is_terminal:
                break
            train_counter += 1
            episode_local_counter += 1
            accum_reward += reward
            if sample_from_replay: # sample minibatch from D
                D.append(state, action, reward, next_state, is_terminal)
                if train_counter > batch_size: # train only if we have at least batch_size samples in D
                    samples = D.sample(batch_size)
                    loss = _train_on_samples(model, samples)
            else: # on-policy
                samples = [Sample(state, action, reward, next_state, is_terminal)]
                loss = _train_on_samples(model, samples)
            print 'each step', time.time() - _tt
            if train_counter % target_reset_freq == 0:
                model.resetTarget()
        print 'episode {0}:\ttrained for {1} steps, accum_reward: {2}'.format(ep, episode_local_counter, accum_reward)


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
    print '_train_on_samples', time.time() - _t
    return loss

train()
