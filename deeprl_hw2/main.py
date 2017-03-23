from atari_environment import AtariEnv
from replay_memory import Sample, ReplayMemory
from models.linear_qn import LinearQN
from models.deep_qn import DeepQN

import time
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser('main driver')
parser.add_argument('--debug', default=False, action='store_true', help='whether use debug mode, shrink initial_buffer, eval_freq, eval_num_episode')
parser.add_argument('--render', default=False, action='store_true', help='whether to render')
parser.add_argument('--model', required=True, help='choose from [linear_qn, linear_double_qn, dqn, double_dqn, duel')
args = parser.parse_args()
print args

# Training parameters
epsilon_init = 1.0
epsilon_final = 0.01
epsilon_decay_steps = 1000000
epsilon_step = (epsilon_final-epsilon_init)/epsilon_decay_steps

batch_size = 32

# Training periods
n_train = 5000000
replay_size = 1000000
if args.debug:
    initial_buffer = 50
else:
    initial_buffer = 50000
target_reset_freq = 10000
model_save_freq = 100000

if args.debug:
    eval_freq = 1000
    eval_num_episode = 2
else:
    eval_freq = 10000
    eval_num_episode = 20

# Create environent and model
env_name = 'SpaceInvaders-v0'
do_render = args.render
env = AtariEnv(env_name, do_render=do_render)

model_name = args.model
if model_name == 'linear_qn':
    fix_target = True
    model = LinearQN(fixTarget=fix_target)
elif model_name == 'dqn':
    double_network = False
    model = DeepQN(doubleNetwork=double_network)
elif model_name == 'double_dqn':
    double_network = True
    model = DeepQN(doubleNetwork=double_network)
else:
    assert False, 'not supported'

sample_from_replay = True # False for Q2
D = ReplayMemory(replay_size)


def evaluate(epsilon):
    sess = model.session
    rewards, steps = [], []
    for ep in range(eval_num_episode):
        state, _, is_terminal = env.new_game()
        accum_reward = 0.
        step = 0
        while not is_terminal:
            if random.random() < epsilon: # uniform_random
                action = env.random_action()
            else: # get action from qn
                action = sess.run(model.next_action, {
                    model.curr_state: state / 255.
                })[0]
            next_state, reward, is_terminal = env.step(action)
            step += 1
            accum_reward += reward
        rewards.append(accum_reward)
        steps.append(step)
        print '[eval][{0}/{1}] accum_reward: {2} after {3} steps'.format(ep+1, eval_num_episode, accum_reward, step)
    avg_reward = sum(rewards) / eval_num_episode
    avg_step = sum(steps) / eval_num_episode
    print '\033[0;31m[eval][eps={0}]\033[0m average_reward: {1} average_step: {2}'.format(epsilon, avg_reward, avg_step)


def train():
    sess = model.session
    train_counter = 0
    ep = 0
    epsilon = epsilon_init
    next_eval_iter = eval_freq
    while train_counter < n_train:
        # Within an episode
        episode_local_counter = 0
        step_time = 0.
        total_loss = 0.
        accum_reward = 0

        if train_counter > next_eval_iter:
            next_eval_iter += eval_freq
            for eps in [epsilon, 0]:
                print 'running eval after train_iter', train_counter, 'with epsilon =', eps
                evaluate(eps)

        state, _, _ = env.new_game()
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
                action = sess.run(model.next_action, {
                    model.curr_state: state / 255.
                })[0]
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

            step_time += time.time()-_tt
            total_loss += loss

            if train_counter % target_reset_freq == 0:
                model.resetTarget()
            
            if train_counter % model_save_freq == 0:
                model.saveModel()
        
        if episode_local_counter == 0:
            print '[buffer] current buffer size: %d' % len(D)
        else:
            ep += 1
            print '[train][{0}] episode {1}: {2} steps, accum_reward: {3}, loss: {4}, epsilon: {5}'.format(train_counter, ep, episode_local_counter, accum_reward, total_loss/episode_local_counter, epsilon)
            print '===== average step_time: %f'%(step_time/episode_local_counter)


def _train_on_samples(model, samples):
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
    return loss

train()
