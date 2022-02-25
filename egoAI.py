from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.networks import Network,encoding_network
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.networks import utils


import gameEnv

env = gameEnv.GomokuEnv(board_size=16)
# env = FlattenActionWrapper(env)
env = tf_py_environment.TFPyEnvironment(env)



# class bakamono_no1(dqn_agent.DqnAgent):
    
#     def __init__(self,time_step_spec,action_spec,observation_spec,board_size=16):
#         self.board_size= board_size
#         self._action_spec = action_spec
#         self._time_step_spec = time_step_spec
#         self._observation_spec = observation_spec
#         print(self._observation_spec)
#         q_net = self.create_q_net()
#         opt = keras.optimizers.Adam()
#         loss = common.element_wise_squared_loss
#         flat_action_spec = tf.nest.flatten(action_spec)
        
#         super().__init__(time_step_spec,action_spec=action_spec,q_network=q_net,optimizer=opt,td_errors_loss_fn=loss)


#     def create_q_net(self):
        
#         q_net = ActorNetwork(self._observation_spec,self.action_spec)
#         return q_net

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

        conv_layers = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8,(3,3),(1,1),activation='relu'),
                                            tf.keras.layers.Conv2D(16,(3,3),(1,1),activation='relu'),
                                            tf.keras.layers.Conv2D(32,(3,3),(1,1),activation='relu'),
                                            tf.keras.layers.Flatten()]) # flatten() -> (batch,3200)
        # conv_layers.build(input_shape=(1,16,16,3))
        pre_combiner_layers = None
        if pre_layers:
            self.pre_layers = pre_layers
        else:
            self.pre_layers = conv_layers
        if pre_combiner:
            self.pre_combiner = pre_combiner
        else:
            self.pre_combiner = pre_combiner_layers

        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]



        self._encoder = encoding_network.EncodingNetwork(
                        observation_spec,
                        preprocessing_layers=self.pre_layers,
                        preprocessing_combiner=self.pre_combiner,
                        conv_layer_params=conv_layer_params,
                        fc_layer_params=fc_layer_params,
                        activation_fn=activation,
                        batch_squash=False)
        self._action_layer = tf.keras.layers.Dense(
                        1,
                        activation=tf.keras.activations.tanh,
                        name='action')
        # self._encoder.create_variables(observation_spec)
        print(self._encoder.summary())
        
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


q_net = ActorNetwork(observation_spec=env.observation_spec(),action_spec=env.action_spec())
# test = bakamono_no1(time_step_spec=env.time_step_spec(),action_spec=env.action_spec(),observation_spec=env.observation_spec())
# print(env.observation_spec())
# print(env.time_step_spec()[-1])
# time_step = env.reset()
# action = test(time_step.observation,time_step.step_type)
# print(action)
# print(env.action_spec())
q_net.build(input_shape=(16,16,3))
print(q_net.summary())
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=tf.keras.optimizers.Adam(),
    td_errors_loss_fn=common.element_wise_squared_loss
)
agent.initialize()