from atari_environment import AtariEnv
from replay_memory import Sample, ReplayMemory

from shallow.linear_qn import LinearQN
from shallow.deep_qn import DeepQN
from shallow.duel_dqn import DuelDQN

import os
import time
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser('main driver')
parser.add_argument('--model', required=True, help='choose from [linear_qn, linear_double_qn, dqn, double_dqn, duel]')
parser.add_argument('--lr', default=0.00025, type=float, help='specfy learning rate, default=0.00025')
parser.add_argument('--eval', default=False, action='store_true', help='whether to evaluate only')
parser.add_argument('--video', default=False, action='store_true', help='whether to produce video [only use with --eval]')
parser.add_argument('--model_dir', default='same', help='directory to model file to use for producing video capture [only use with --eval --video]')
parser.add_argument('--debug', default=False, action='store_true', help='whether use debug mode, shrink initial_buffer, eval_freq, eval_num_episode')
parser.add_argument('--render', default=False, action='store_true', help='whether to render')
args = parser.parse_args()
print args

# Training parameters
epsilon_init = 1.0
epsilon_final = 0.1
epsilon_decay_steps = 1000000
epsilon_step = (epsilon_final-epsilon_init)/epsilon_decay_steps

batch_size = 32

# Training periods
n_train = 5000000
replay_size = 1000000
target_reset_freq = 10000
model_save_freq = 100000
if args.debug:
    initial_buffer = 50
else:
    initial_buffer = 50000

if args.debug:
    eval_freq = 1000
    eval_num_episode = 2
else:
    eval_freq = 50000
    eval_num_episode = 20
    
print 'learning rate', args.lr

model_name = args.model
if args.model_dir == 'same':
    model_dir = model_name
else:
    model_dir, model_iter = args.model_dir.split('.')[0].rsplit('/', 1)
    assert os.path.exists(model_dir)
    cmd = 'echo \'model_checkpoint_path: "{0}"\' > {1}'.format(model_iter, os.path.join(model_dir, 'checkpoint'))
    print cmd
    os.system(cmd)

if model_name == 'linear_qn':
    model = LinearQN(model_dir=model_dir, fixTarget=True, doubleNetwork=False, lr=args.lr)
elif model_name == 'linear_double_qn':
    model = LinearQN(model_dir=model_dir, fixTarget=True, doubleNetwork=True, lr=args.lr)
elif model_name == 'dqn':
    model = DeepQN(model_dir=model_dir, doubleNetwork=False, lr=args.lr)
elif model_name == 'double_dqn':
    model = DeepQN(model_dir=model_dir, doubleNetwork=True, lr=args.lr)
elif model_name == 'duel_dqn':
    model = DuelDQN(model_dir=model_dir, lr=args.lr)
else:
    assert False, 'not supported'

env_name = 'SpaceInvaders-v0'
do_render = args.render
env = AtariEnv(env_name, model_name, do_render=do_render)

sample_from_replay = True # False for Q2
D = ReplayMemory(replay_size)


def evaluate(epsilon):
    sess = model.session
    rewards, scores, steps = [], [], []
    for ep in range(eval_num_episode):
        state, _, is_terminal = env.new_game()
        accum_reward = 0.
        total_score = 0.
        step = 0
        while not is_terminal:
            if random.random() < epsilon: # uniform_random
                action = env.random_action()
            else: # get action from qn
                action = sess.run(model.next_action, {
                    model.curr_state: state / 255.
                })[0]
            state, reward, is_terminal, score = env.step(action, include_noclip=True)
            step += 1
            accum_reward += reward
            total_score += score
        rewards.append(accum_reward)
        scores.append(total_score)
        steps.append(step)
        print '[eval][{0}/{1}] accum_reward: {2} total_score: {3} after {4} steps'.format(ep+1, eval_num_episode, accum_reward, total_score, step)
    avg_reward = sum(rewards) / eval_num_episode
    avg_score = sum(scores) / eval_num_episode
    avg_step = sum(steps) / eval_num_episode
    print '\033[0;31m[eval][eps={0}]\033[0m average_reward: {1} average_score: {2} average_step: {3}'.format(epsilon, avg_reward, avg_score, avg_step)

def eval_only(make_video):
    global env
    env = AtariEnv(env_name, model_name, do_render=do_render, make_video=make_video)
    sess = model.session
    train_counter = sess.run(model.global_step)
    # for eps in [1., 0.05, 0.]:
    for eps in [0.05]:
        print 'running eval after train_iter', train_counter, 'with epsilon =', eps
        evaluate(eps)
        
def train():
    sess = model.session
    train_counter = sess.run(model.global_step)
    ep = 0
    epsilon = max(epsilon_final, epsilon_init + epsilon_step*train_counter)
    next_eval_iter = eval_freq * (train_counter/eval_freq + 1)

    reset_op = model.resetTarget()
    if reset_op:
        sess.run(reset_op)
    
    while train_counter < n_train:
        # Within an episode
        episode_local_counter = 0
        step_time = 0.
        total_loss = 0.
        accum_reward = 0

        if train_counter > next_eval_iter:
            next_eval_iter += eval_freq
            eps = 0.05
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
                state = next_state
                continue
            
            epsilon = max(epsilon_final,epsilon+epsilon_step)
            
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

            state = next_state

            step_time += time.time()-_tt
            total_loss += loss

            if train_counter % target_reset_freq == 0:
                reset_op = model.resetTarget()
                if reset_op:
                    sess.run(reset_op)
            
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

if args.eval:
    eval_num_episode = 500
    eval_only(make_video=args.video)
else:
    train()
