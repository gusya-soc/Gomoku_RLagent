from tf_agents.policies import random_tf_policy
from pygame.locals import *
from GameMap import *
from ChessAI import *
from tf_agents.specs import tensor_spec
from egoAI import ActorNetwork as net
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
import reverb
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common
from gameEnv import GomokuEnv
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
import tensorflow as tf
from tf_agents.networks import value_network, actor_distribution_network
from tf_agents.agents import ReinforceAgent
import tf_agents
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tqdm import tqdm
from tf_agents.agents import DqnAgent
from tf_agents.networks import sequential

BOARD_SIZE = 15


class TrainGame():
    def __init__(self, caption, play_mode, AI_first):

        self.mode = play_mode
        self.is_play = False

        self.map = Map(CHESS_LEN, CHESS_LEN)
        self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
        self.action = None
        self.AI = ChessAI(CHESS_LEN)
        self.AI_first = AI_first
        self.winner = None

    def start(self):
        self.is_play = True
        self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
        self.map.reset()
        self.AI.number = 0
        self.useAI = True
        self.winner = None

    def play(self):
        if self.is_play and not self.isOver():
            if self.useAI:
                x, y = self.AI.findBestChess(self.map.map, self.player)
                self.checkClick(x, y, True)
                self.useAI = False

        if self.isOver():
            # self.showWinner()
            pass

    def checkClick(self, x, y, isAI=False):
        self.AI.click(self.map, x, y, self.player)
        # print(self.map.map,x,y,self.player)
        if self.AI.isWin(self.map.map, self.player):
            self.winner = self.player
        else:
            self.player = self.map.reverseTurn(self.player)
            if not isAI and self.mode != USER_VS_USER_MODE:
                self.useAI = True

    def isOver(self):
        return self.winner is not None

    def agent_action(self):
        pass


game = TrainGame(None, 3, True)


class TrainEnv(GomokuEnv):
    def __init__(self):
        super().__init__()
        self.game = game

        # init_step =
        # 开始一个Game，AI下一步棋。返回初始棋盘。
        # the trian step =
        # Action下一步棋，将坐标返回给gameAI再下一步棋，更新棋盘。返回该棋盘的observation

    def _reset(self):

        self.game.start()
        self.game.play()
        map = np.array(self.game.map.map).T
        step = np.array(self.game.map.steps)
        self.update_state(map, step)
        self._is_ended = False
        print("="*80)
        # print("was reseted")
        return ts.restart(self._observation)

    def _step(self, action):
        if self._is_ended:
            return self.reset()
        print(action)
        action = self.action_to_coordinate(action)
        action = net.extend_action(action)

        x = action[0]
        y = action[1]
        print(f"{x},{y}")
        if self._empty_state[x][y] == 0:
            print('abort empty')
            reward = -100.
            self._is_ended = True
            return ts.termination(self._observation, reward)

        self.game.checkClick(x, y, False)
        if self.game.winner == MAP_ENTRY_TYPE.MAP_PLAYER_TWO:
            reward = 100.
            print("win")
            self._is_ended = True
            return ts.termination(self._observation, reward)

        self.game.play()
        map = np.array(self.game.map.map).T
        step = np.array(self.game.map.steps)
        self.update_state(map, step)
        if self.game.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
            reward = -100.
            print("loss")
            self._is_ended = True
            return ts.termination(self._observation, reward)
        # print(np.array(self.game.map.map))
        reward = 100.
        return ts.transition(self._observation, reward)

# while True:
#     game.play()
#     pygame.display.update()


#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()
#         elif event.type == pygame.MOUSEBUTTONDOWN:
#             mouse_x, mouse_y = pygame.mouse.get_pos()
#             game.mouseClick(mouse_x, mouse_y)
#             game.check_buttons(mouse_x, mouse_y)
num_iterations = 250  # @param {type:"integer"}
collect_episodes_per_iteration = 50  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}

fc_layer_params = (100,)

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 10  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 50


env = TrainEnv()
time_step = env.reset()
action = np.array(1, dtype=np.float32)
next_time_step = env.step(action)

train_py_env = env
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(train_py_env)

fc_layer_params = (100,)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
flatten = tf.keras.layers.Flatten()
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation='relu',
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential([flatten]+dense_layers + [q_values_layer])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

tf_agent = DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

tf_agent.initialize()
print(q_net.summary())
eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

compute_avg_return(eval_env, random_policy, num_eval_episodes)

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=2000,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=100).run(train_py_env.reset())

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=16,
    num_steps=2).prefetch(3)

iterator = iter(dataset)
next(iterator)

agent = tf_agent
# # (Optional) Optimize by wrapping some of the code in a graph using TF function.
# agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=20)

for _ in range(num_iterations):

    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)
    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
