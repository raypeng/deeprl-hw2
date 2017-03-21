from atari_environment import AtariEnv
from replay_memory import ReplayMemory
# from deep_q import DeepQ



gamma = 0.99
lr = 1e-4
epsilon = 0.05
n_train = 5000000
replay_size = 1000000
target_reset_freq = 10000
batch_size = 32
M = 200
T = 10000

env_name = 'SpaceInvaders-v0'
env = AtariEnv(env_name)
model = DeepQ()

sample_from_replay = True # False for Q2
if sample_from_replay:
    D = ReplayMemory(replay_size)

def train():
    sess = model.session
    train_counter = 0
    for ep in range(M):
        state, _, _ = env.new_game()
        while true:
            if random.random() < epsilon: # uniform_random
                action = env.random_action()
            else: # get action from qn
                action_tensor = model.get_action()
                action = sess.run([action_tensor], {
                    model.single_state_input: state
                })
            next_state, reward, is_terminal = env.step(action)
            if is_terminal:
                break
            if sample_from_replay: # sample minibatch from D
                D.append(state, action, reward, next_state, is_terminal)
                if t > batch_size: # train only if we have at least batch_size samples in D
                    samples = D.sample(batch_size)
                    _train_on_samples(model, samples)
                    train_counter += 1
            else: # on-policy
                samples = [state, action, reward, next_state, is_terminal]
                _train_on_samples(model, samples)
                train_counter += 1
            if train_counter % target_reset_freq == 0:
                model.resetTarget()

def _train_on_samples(model, samples):
    sess = model.session
    state_list = np.array([s.state for s in samples])
    action_list = np.array([s.action for s in samples])
    reward_list = np.array([s.reward for s in samples])
    next_state_list = np.array([s.next_state for s in samples])
    is_terminal_list = np.array([s.is_terminal for s in samples]) + 0. # True -> 1
    _, loss = sess.run([model.train_op, model.loss], {
        model.state_input: state_list,
        model.action_input: action_list,
        model.reward_input: reward_list,
        model.nextState_input: next_state_list,
        model.isTerminal_input: is_terminal_list
    })
    return loss
