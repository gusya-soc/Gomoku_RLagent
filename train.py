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
from tf_agents.networks import value_network
from tf_agents.agents import ReinforceAgent
import tf_agents

BOARD_SIZE = 16


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
        map = np.array(self.game.map.map)
        step = np.array(self.game.map.steps)
        self.update_state(map, step)

        self._rewards = 0.

        return ts.restart(self._observation)

    def _step(self, action):
        action = net.extend_action(action)
        x = action[0]
        y = action[1]

        if self._empty_state[x][y] == 0:
            self._rewards = -10000.
            return ts.termination(self._observation, self._rewards)

        self.game.checkClick(x, y, False)
        self._rewards -= 1.
        if self.game.winner == MAP_ENTRY_TYPE.MAP_PLAYER_TWO:
            self._rewards += 100.
            return ts.termination(self._observation, self._rewards)

        self.game.play()
        map = np.array(self.game.map.map)
        step = np.array(self.game.map.steps)
        self.update_state(map, step)
        if self.game.winner == MAP_ENTRY_TYPE.MAP_PLAYER_ONE:
            self._rewards -= 100.
            return ts.termination(self._observation, self._rewards)

        return ts.transition(self._observation, self._rewards)

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

env = TrainEnv()
env = tf_py_environment.TFPyEnvironment(env)
conv_filters = 8
k_size = (3,3)
stride = (1,1)
conv_params = [(conv_filters,k_size,stride),(conv_filters*2,k_size,stride),(conv_filters*4,k_size,stride)]
conv_layers = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8,(3,3),(1,1),activation='relu'),
                                    tf.keras.layers.Conv2D(16,(3,3),(1,1),activation='relu'),
                                    tf.keras.layers.Conv2D(32,(3,3),(1,1),activation='relu'),
                                    tf.keras.layers.Flatten()])



combiner = tf.keras.layers.Concatenate(axis=-1)


actor_net = net(observation_spec=env.observation_spec(),action_spec=env.action_spec(),pre_layers=conv_layers,pre_combiner=None,fc_layer_params=(64,128,64))
value_net = value_network.ValueNetwork(env.observation_spec(),conv_layer_params=conv_params)
# test = bakamono_no1(time_step_spec=env.time_step_spec(),action_spec=env.action_spec(),observation_spec=env.observation_spec())
# print(env.observation_spec())
# print(env.time_step_spec()[-1])
# time_step = env.reset()
# action = test(time_step.observation,time_step.step_type)
# print(action)
# print(env.action_spec())
value_net.create_variables(env.observation_spec())
actor_net.create_variables(env.observation_spec())
print(actor_net.summary())
print(value_net.summary())
# agent = dqn_agent.DqnAgent(
#     env.time_step_spec(),
#     env.action_spec(),
#     q_network=q_net,
#     optimizer=tf.keras.optimizers.Adam(),
#     td_errors_loss_fn=common.element_wise_squared_loss
# )
# agent.initialize()
# agent.policy
agent = ReinforceAgent(time_step_spec=env.time_step_spec(),
                        action_spec=env.action_spec(),
                        actor_network=actor_net,
                        optimizer=tf.keras.optimizers.Adam(),
                        )
agent.initialize()
print(agent.policy)
policy = agent.policy
time_step = env.reset()
action = policy.action(time_step)
print(action)
time_step = env.step(action)
action = policy.action(time_step)
print(list(action[0].numpy()[0].astype('int32')))
# print(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
print(replay_buffer_signature)
print(action)
print(action.action.numpy())


table_name = 'uniform_table'
replay_buffer_capacity = 2000
replay_buffer_signature = tensor_spec.from_spec(
    agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)
reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)

random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())

driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
    [rb_observer],
    max_episodes=2)

print(driver)
driver.run(env.reset())


def collect_episode(environment, policy, num_episodes):

    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
        [rb_observer],
        max_episodes=num_episodes)
    initial_time_step = environment.reset()

    driver.run(initial_time_step)


num_iterations = 250  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 50
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.


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
    return avg_return


avg_return = compute_avg_return(env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        env, agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = agent.train(experience=trajectories)

    replay_buffer.clear()

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            game.env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
