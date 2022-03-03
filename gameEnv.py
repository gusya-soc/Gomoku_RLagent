import abc
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import TimeStep,StepType
from tf_agents.environments.wrappers import FlattenActionWrapper
from ChessAI import GAME_PLAY_MODE,AI_RUN_FIRST
from train import TrainGame as Game

game = Game("FIVE CHESS ", GAME_PLAY_MODE, AI_RUN_FIRST)
class GomokuEnv(py_environment.PyEnvironment):

    def __init__(self,board_size=16):
        super().__init__()
        self.board_size = board_size
        self._action = array_spec.BoundedArraySpec(shape=(2,),dtype=np.float32,minimum=0,maximum=16,name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(board_size,board_size,3),maximum=1,dtype=np.float32,name='observation')
        # self._rewards_spec = array_spec.ArraySpec(shape=(1,),dtype=np.float32,name='reward')
        # self._current_state = np.zeros(shape=(board_size,board_size),dtype=np.float32)
        # self._oppo_state = np.zeros(shape=(board_size,board_size),dtype=np.float32)
        # self._empty_state = np.zeros(shape=(board_size,board_size),dtype=np.float32)
        # self._rewards = 0.
        # self._observation_spec = {'state':array_spec.ArraySpec(shape=(self.board_size,self.board_size,3),dtype=np.float32),
        #                             'vector':array_spec.BoundedArraySpec((5,),np.float32,minimum=0,maximum=1)}

        
                            


    def action_spec(self):
        return self._action

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_state = np.zeros(shape=(self.board_size,self.board_size,1),dtype=np.float32)
        self._oppo_state = np.zeros(shape=(self.board_size,self.board_size,1),dtype=np.float32)
        self._empty_state = np.zeros(shape=(self.board_size,self.board_size,1),dtype=np.float32)
        self._observation = np.concatenate([self._current_state,self._oppo_state,self._empty_state],axis=2,dtype=np.float32)
        # self._rewards = np.zeros((1,),dtype=np.float32)
        # self._ob = {'current_state':np.zeros(shape=(self.board_size,self.board_size,1),dtype=np.float32),
        #             'oppo_state':np.zeros(shape=(self.board_size,self.board_size,1),dtype=np.float32),
        #             'empty_state':np.zeros(shape=(self.board_size,self.board_size,1),dtype=np.float32)}
        self._rewards = 0.
        return ts.restart(self._observation)        


    def split_map(self,map,where):
        map = np.where(map==where,1).reshape(self.board_size,self.board_size,1)
        return map
    def update_state(self,map,steps): 
        self._oppo_state = self.split_map(map,1)
        self._current_state = self.split_map(map,2)
        self._empty_state = self.split_map(map,0)
        self.observation = np.concatenate([self._current_state,self._oppo_state,self._empty_state],axis=2,dtype=np.float32)


        # TODO
        self._last_step = np.zeros((self.board_size,self.board_size,1))
        step_x = steps[-1][0]
        step_y = steps[-1][1]
        self._last_step[step_x][step_y] = 1
    def _step(self,action):

        # self._current_state[:] = 1
        # print(self._current_state.shape)
        # time_step = ts.TimeStep(step_type=ts.StepType.MID,reward=self._rewards,discount=1.,observation=self._current_state)
        # return time_step
        # print(self._observation)
        # print(self._rewards)
        if self._rewards >= 4:
            return ts.termination(
                self._observation,
                self._rewards
            )
        self._rewards +=1
        return ts.transition(self._observation,self._rewards,discount=1.0)
        #     #     self._rewards += 1
        # #     return TimeStep(step_type=StepType.FIRST,reward=self._rewards,discount=1.0,observation=self._ob)
        # return TimeStep(step_type=StepType.LAST,reward=self._rewards,discount=1.0,observation=self._ob)


class TrainEnv(GomokuEnv):
    def __init__(self):
        super().__init__()


        #init_step = 
        #开始一个Game，AI下一步棋。返回初始棋盘。
        #the trian step = 
        #Action下一步棋，将坐标返回给gameAI再下一步棋，更新棋盘。返回该棋盘的observation
    def _reset(self):
        self.game.start()
        self.game.play()
        map = self.game.map.map
        step = self.game.map.steps
        self.update_state(map,step)
    
    def _step(self, action):
        pass
        

# test = GomokuEnv()
# print(test.time_step_spec())
# print(test.reset())
# test = tf_py_environment.TFPyEnvironment(test)
# print(test.time_step_spec())

# time_step = test.reset()
# time_step = test.step(np.array(1))
# time_step = test.step(np.array(1))
# time_step = test.step(np.array(1))
# time_step = test.step(np.array(1))
# # print(time_step._fields)

# print(test.observation_spec())

# # # print(test.time_step_spec())
# # print(time_step.observation['empty_state'])
# print(time_step.reward)
# print(time_step.observation)
# print(test.time_step_spec())
# a = utils.validate_py_environment(test,episodes=5)

from egoAI import ActorNetwork
from tf_agents.networks import value_network
from tf_agents.agents import ReinforceAgent
from tf_agents.specs import tensor_spec
env = GomokuEnv()
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


actor_net = ActorNetwork(observation_spec=env.observation_spec(),action_spec=env.action_spec(),pre_layers=conv_layers,pre_combiner=None,fc_layer_params=(64,128,64))
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