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
        self._rewards = np.array([0.])
        # self._rewards = 0.
        return ts.restart(self._observation,reward_spec=self._rewards)        


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
        return ts.transition(
            self._observation,
            self._rewards,
            discount=1.0
        )
        #     self._rewards += 1
        #     return TimeStep(step_type=StepType.FIRST,reward=self._rewards,discount=1.0,observation=self._ob)
        # return TimeStep(step_type=StepType.LAST,reward=self._rewards,discount=1.0,observation=self._ob)
# test = GomokuEnv()

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