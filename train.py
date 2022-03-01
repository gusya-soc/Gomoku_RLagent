import pygame
from pygame.locals import *
from GameMap import *
from ChessAI import *
from gameEnv import *
import egoAI
from tf_agents.specs import tensor_spec
from egoAI import ActorNetwork as net
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
import reverb
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common

BOARD_SIZE = 16

class Game():
	def __init__(self, caption, play_mode, AI_first):

		self.mode = play_mode
		self.is_play = False

		self.map = Map(CHESS_LEN, CHESS_LEN)
		self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
		self.action = None
		self.AI = ChessAI(CHESS_LEN)
		self.AI_first = AI_first
		self.winner = None
		self.env = GomokuEnv(board_size=BOARD_SIZE)
		self.agent = egoAI.bakamono_no1(self.env.time_step_spec(),action_spec=self.env.action_spec(),actor_net=egoAI.actor_net,value_net=egoAI.value_net)
		self.agent.initialize()
		self.policy = self.agent.policy
		
	def start(self):
		self.is_play = True
		self.player = MAP_ENTRY_TYPE.MAP_PLAYER_ONE
		self.map.reset()
		self.AI.number = 0
		self.useAI =True
		self.time_step = self.env.reset()

	def play(self):
		if self.is_play and not self.isOver():
			if self.useAI:
				x, y = self.AI.findBestChess(self.map.map, self.player)
				self.checkClick(x, y, True)
				self.useAI = False

			if self.mode ==EGO_VS_AI_MODE and self.useAI == False:
				self.env.update_state(self.map.map,self.map.steps)
				self.time_step = self.env.step(None)
				action = self.policy.action(self.time_step)
				action = net.extend_action(action)
				self.checkClick(action[0],action[1])	


		if self.isOver():
			# self.showWinner()
			pass

	
	
	def checkClick(self,x, y, isAI=False):
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


			
game = Game("FIVE CHESS " + GAME_VERSION, GAME_PLAY_MODE, AI_RUN_FIRST)
tf_agent = game.agent
# while True:
# 	game.play()
# 	pygame.display.update()
	
# 	for event in pygame.event.get():
# 		if event.type == pygame.QUIT:
# 			pygame.quit()
# 			exit()
# 		elif event.type == pygame.MOUSEBUTTONDOWN:
# 			mouse_x, mouse_y = pygame.mouse.get_pos()
# 			game.mouseClick(mouse_x, mouse_y)
# 			game.check_buttons(mouse_x, mouse_y)

table_name = 'uniform_table'
replay_buffer_capacity = 2000

replay_buffer_signature = tensor_spec.from_spec(
      game.agent.collect_data_spec)
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
    game.agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)

def collect_episode(environment, policy, num_episodes):

  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    [rb_observer],
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)


num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}
learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

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
  return avg_return.numpy()[0]
avg_return = compute_avg_return(game.env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few episodes using collect_policy and save to the replay buffer.
  collect_episode(
      game.env, tf_agent.collect_policy, collect_episodes_per_iteration)

  # Use data from the buffer and update the agent's network.
  iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
  trajectories, _ = next(iterator)
  train_loss = tf_agent.train(experience=trajectories)  

  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)