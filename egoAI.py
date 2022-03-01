from re import A
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents import ReinforceAgent
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.networks import Network,encoding_network,value_network
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec


import gameEnv

env = gameEnv.GomokuEnv(board_size=16)
# env = FlattenActionWrapper(env)
env = tf_py_environment.TFPyEnvironment(env)



class bakamono_no1(ReinforceAgent):
    
    def __init__(self,time_step_spec,action_spec,actor_net,value_net):
        # self._action_spec = action_spec
        # self._time_step_spec = time_step_spec
        # self._actor_network = actor_net
        
        
        opt = tf.keras.optimizers.Adam()
        super().__init__(time_step_spec,action_spec=action_spec,actor_network=actor_net,optimizer=opt,value_network=value_net)
        
        # agent = ReinforceAgent(time_step_spec=env.time_step_spec(),
        #                 action_spec=env.action_spec(),
        #                 actor_network=actor_net,
        #                 optimizer=tf.keras.optimizers.Adam(),
        #                 )

#     def create_q_net(self):
        
#         q_net = ActorNetwork(self._observation_spec,self.action_spec)
#         return q_net
# conv_layers = {'state':tf.keras.models.Sequential([tf.keras.layers.Conv2D(8,(3,3),(1,1),activation='relu'),
#                                     tf.keras.layers.Conv2D(16,(3,3),(1,1),activation='relu'),
#                                     tf.keras.layers.Conv2D(32,(3,3),(1,1),activation='relu'),
#                                     tf.keras.layers.Flatten()]),
                # 'vector':tf.keras.layers.Dense(5)} # flatten() -> (batch,3200)
conv_filters = 8
k_size = (3,3)
stride = (1,1)
conv_params = [(conv_filters,k_size,stride),(conv_filters*2,k_size,stride),(conv_filters*4,k_size,stride)]
conv_layers = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8,(3,3),(1,1),activation='relu'),
                                    tf.keras.layers.Conv2D(16,(3,3),(1,1),activation='relu'),
                                    tf.keras.layers.Conv2D(32,(3,3),(1,1),activation='relu'),
                                    tf.keras.layers.Flatten()])



combiner = tf.keras.layers.Concatenate(axis=-1)




class ActorNetwork(Network):
    def __init__(self,observation_spec,
                    action_spec,
                    pre_layers=None,
                    pre_combiner=None,
                    conv_layer_params=None,
                    fc_layer_params=None,
                    activation=tf.keras.activations.relu,
                    name='ActorNetwork'):
                    
        super().__init__(input_tensor_spec=observation_spec,state_spec=(),name=name)


        # conv_layers.build(input_shape=(1,16,16,3))

        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError('Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]
        if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
            raise ValueError('Only float actions are supported by this network.')



        self._encoder = encoding_network.EncodingNetwork(
                        observation_spec,
                        preprocessing_layers=pre_layers,
                        preprocessing_combiner=pre_combiner,
                        conv_layer_params=conv_layer_params,
                        fc_layer_params=fc_layer_params,
                        activation_fn=activation,
                        batch_squash=False)
        self._action_layer = tf.keras.layers.Dense(
                        2,
                        activation=tf.keras.activations.tanh,
                        name='action')
        # self._encoder.create_variables(observation_spec)
        # print(self._encoder.summary())
        
    def call(self,observations,step_type=(),network_state=(),**kwargs):
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        # 顺序，pre_layers > pre_combiner > post_layers(fc-128-64) > self._action_layer)

        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)
        # print(observations)
        state, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state)
        actions = self._action_layer(state)
        actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        # print(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state

    def create_pre_layers(self):

        return 0

    def create_pre_combiner(self):
        return tf.keras.layers.Concatenate(axis=-1)
    
    @staticmethod
    def extend_action(action):
        action = action[0].numpy()[0].astype('int32')
        action = list(action)
        return action



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
ts = env.reset()
action = policy.action(ts)
print(action)
ts = env.step(action)
action = policy.action(ts)
print(list(action[0].numpy()[0].astype('int32')))
# print(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
print(replay_buffer_signature)