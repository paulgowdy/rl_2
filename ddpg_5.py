import pickle
import numpy as np
import time

from osim.env import ProstheticsEnv as RunEnv

from rpm import rpm # import threading

import tensorflow as tf
import canton as ct
from canton import *

import matplotlib.pyplot as plt 

from process_dict_obs import *
from reward_shaping import *

class ddpg_agent(object):

	def __init__(self,
		observation_space_dims,
		discount_factor,
		nb_actions = 19,
		rpm_size = 1500000
		



		):

		self.training = True
		self.discount_factor = discount_factor
		#self.noise_source = one_fsq_noise()
		#self.train_counter = 0
		self.train_multiplier = 1



		self.rpm = rpm(rpm_size)

		# Deal only with the continuous space for now...
		self.inputdims = observation_space_dims
		self.outputdims = nb_actions

		def clamper(actions):
			
			return np.clip(actions, a_max = 1.0 , a_min = 0.0)

		self.clamper = clamper

		ids, ods = self.inputdims, self.outputdims

		self.actor = self.create_actor_network(ids,ods)
		self.critic = self.create_critic_network(ids,ods)
		self.actor_target = self.create_actor_network(ids,ods)
		self.critic_target = self.create_critic_network(ids,ods)

		
		self.feed, self.joint_inference, sync_target = self.train_step_gen()

		sess = ct.get_session()
		sess.run(tf.global_variables_initializer())

		sync_target()

		import threading as th
		self.lock = th.Lock()

		self.reward_plotter = plt.figure()
		self.reward_collector = []

		self.phased_noise_anneal_duration = 100
		
	# a = actor(s) : predict actions given state
	def create_actor_network(self,inputdims,outputdims):
		
		# add gaussian noise... to params?

		rect = Act('selu')
		magic = 1

		def d(i,o):
			return LayerNormDense(i,o,stddev=magic)
			

		c = Can()
		
		c.add(d(inputdims,800))
		c.add(rect)
		
		c.add(d(800,400))
		c.add(rect)
		
		c.add(Dense(400,outputdims,stddev=1))

		
		c.add(Act('tanh'))
		c.add(Lambda(lambda x: x*0.5 + 0.5))
	

		c.chain()

		return c

	# q = critic(s,a) : predict q given state and action
	def create_critic_network(self,inputdims,actiondims):
	
		rect = Act('selu')
		magic = 1

		def d(i,o):
			return LayerNormDense(i,o,stddev=magic)
			

		c = Can()

		concat = Lambda(lambda x:tf.concat(x,axis=1))

		
		den0 = c.add(d(inputdims, 800))
		
		den1 = c.add(d(800 + actiondims, 400))
		den2 = c.add(d(400,200))
		den2a = c.add(d(200,100))
		den3 = c.add(Dense(100,1,stddev=1))

		def call(i):
			state = i[0]
			action = i[1]
			
			i = (rect((den0(state))))
			i = concat([i,action])
			i = (rect((den1(i))))
			i = (rect((den2(i))))
			i = (rect((den2a(i))))
			i = den3(i)
			q = i
			return q

		c.set_function(call)

		return c

	def train_step_gen(self):
		s1 = tf.placeholder(tf.float32,shape=[None,self.inputdims])
		a1 = tf.placeholder(tf.float32,shape=[None,self.outputdims])
		r1 = tf.placeholder(tf.float32,shape=[None,1])
		isdone = tf.placeholder(tf.float32,shape=[None,1])
		s2 = tf.placeholder(tf.float32,shape=[None,self.inputdims])

		# 1. update the critic
		a2 = self.actor_target(s2)
		q2 = self.critic_target([s2,a2])
		q1_target = r1 + (1-isdone) * self.discount_factor * q2
		q1_predict = self.critic([s1,a1])
		critic_loss = tf.reduce_mean((q1_target - q1_predict)**2)
		# produce better prediction

		# # # huber loss per zzz
		# diff = q1_target - q1_predict
		# abs_diff = tf.abs(diff)
		# sqr_diff = tf.square(diff)
		# clipper = 1.0
		# condition = tf.to_float(abs_diff < clipper)
		# sqr_loss = 0.5 * sqr_diff
		# linear_loss = clipper * (abs_diff - 0.5 * clipper)
		# critic_loss = sqr_loss * condition + linear_loss * (1.0 - condition)
		# critic_loss = tf.reduce_mean(critic_loss)

		# 2. update the actor
		a1_predict = self.actor(s1)
		q1_predict = self.critic([s1,a1_predict])
		actor_loss = tf.reduce_mean(- q1_predict)
		# maximize q1_predict -> better actor

		# 3. shift the weights (aka target network)
		tau = tf.Variable(1e-3) # original paper: 1e-3. need more stabilization
		aw = self.actor.get_weights()
		atw = self.actor_target.get_weights()
		cw = self.critic.get_weights()
		ctw = self.critic_target.get_weights()

		one_m_tau = 1-tau

		shift1 = [tf.assign(atw[i], aw[i]*tau + atw[i]*(one_m_tau))
			for i,_ in enumerate(aw)]
		shift2 = [tf.assign(ctw[i], cw[i]*tau + ctw[i]*(one_m_tau))
			for i,_ in enumerate(cw)]

		# 4. inference
		set_training_state(False)
		a_infer = self.actor(s1)
		q_infer = self.critic([s1,a_infer])
		set_training_state(True)

		# 5. L2 weight decay on critic
		decay_c = tf.reduce_sum([tf.reduce_sum(w**2) for w in cw])* 1e-7
		decay_a = tf.reduce_sum([tf.reduce_sum(w**2) for w in aw])* 1e-7

		decay_c = 0
		decay_a = 0

		# # optimizer on
		# # actor is harder to stabilize...
		opt_actor = tf.train.AdamOptimizer(1e-4)
		opt_critic = tf.train.AdamOptimizer(3e-4)
		# # opt_actor = tf.train.RMSPropOptimizer(1e-3)
		# # opt_critic = tf.train.RMSPropOptimizer(1e-3)
		opt = tf.train.AdamOptimizer(3e-5)
		opt_actor,opt_critic = opt,opt
		cstep = opt_critic.minimize(critic_loss+decay_c, var_list=cw)
		astep = opt_actor.minimize(actor_loss+decay_a, var_list=aw)

		self.feedcounter=0

		def feed(memory):

			[s1d,a1d,r1d,isdoned,s2d] = memory 
			sess = ct.get_session()
			res = sess.run([critic_loss,actor_loss,
				cstep,astep,shift1,shift2],
				feed_dict={
				s1:s1d,a1:a1d,r1:r1d,isdone:isdoned,s2:s2d,tau:1e-3
				})

			#debug purposes
			self.feedcounter+=1
			if self.feedcounter%10==0:
				print(' '*30, 'closs: {:6.4f} aloss: {:6.4f}'.format(
					res[0],res[1]),end='\r')


		def joint_inference(state):

			sess = ct.get_session()
			res = sess.run([a_infer,q_infer],feed_dict={s1:state})
			return res

		def sync_target():
			sess = ct.get_session()
			sess.run([shift1,shift2],feed_dict={tau:1.})

		return feed, joint_inference, sync_target	

	def feed_one(self,tup):
		self.rpm.add(tup)

	def play_one_ep(self, env, noise_level, frameskip, max_steps = 50000):

		timer = time.time()

		steps = 0
		total_reward = 0
		total_q = 0
		episode_memory = []

		# Need to add noise to actions...

		# Reset Env
		observation_d = env.reset(project = False)
		observation = process_obs_dict(observation_d)

		noise_phase = int(np.random.uniform()*99999)

		while True and steps <= max_steps:
			
			

			observation_before_action = observation

			action, q = self.act(observation_before_action)
	
			total_q += q

			# ADD NOISE HERE
			
			phased_noise_amplitude = ((- noise_phase - steps) % self.phased_noise_anneal_duration) / self.phased_noise_anneal_duration
            
			exploration_noise = np.random.normal(size=(self.outputdims,)) * noise_level * phased_noise_amplitude


			action += exploration_noise
			action = self.clamper(action) # don't clamp, see what happens.
			action_out = action

			# implement frameskip here

			learning_reward = 0

			for _ in range(frameskip):

				steps +=1
			
				observation_d, reward, done, _info = env.step(action_out, project = False)
				
				total_reward += reward
				learning_reward += reward_shaping(observation_d, reward)

			#learning_reward = reward_shaping(observation_d, learning_reward)

			observation = process_obs_dict(observation_d)

			isdone = 1 if done else 0

			if self.training == True:
		  
				self.feed_one((observation_before_action, action, learning_reward, isdone, observation)) # s1,a1,r1,isdone,s2
				self.train()
				#print('train here')

			if done :
				
				break

		totaltime = time.time()-timer
		print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
		steps,totaltime,totaltime/steps,total_reward
		))

		# PLOT
		self.reward_collector.append(total_reward)

		if len(self.reward_collector) % 10 == 0:

			self.reward_plotter.clf()
			plt.plot(self.reward_collector, c = 'k')
			plt.grid(linestyle = '-.')
			plt.pause(0.001)

		return total_reward

	def act(self, observation):

		#actor,critic = self.actor,self.critic

		obs = np.reshape(observation,(1,len(observation)))

		[actions,q] = self.joint_inference(obs)
	   

		actions,q = actions[0],q[0]

	
		return actions,q

	def train(self, batch_size = 64, min_memory_size = 200):

		memory = self.rpm
		#batch_size = 64
		total_size = batch_size
		epochs = 1

		# self.lock.acquire()
		if memory.size() > min_memory_size:

			#if enough samples in memory
			for i in range(self.train_multiplier):
				# sample randomly a minibatch from memory
				[s1,a1,r1,isdone,s2] = memory.sample_batch(batch_size)
				# print(s1.shape,a1.shape,r1.shape,isdone.shape,s2.shape)

				self.feed([s1,a1,r1,isdone,s2])

		# self.lock.release()

	def save_weights(self, dir_prefix = 'training_saves/weights/'):
		networks = ['actor','critic','actor_target','critic_target']
		for name in networks:
			network = getattr(self,name)
			network.save_weights(dir_prefix + 'ddpg_'+name+'.npz')

	def load_weights(self, dir_prefix = 'training_saves/weights/'):
		networks = ['actor','critic','actor_target','critic_target']
		for name in networks:
			network = getattr(self,name)
			network.load_weights(dir_prefix + 'ddpg_'+name+'.npz')


if __name__=='__main__':

	env = RunEnv(visualize=False)

	env.change_model(model='2D', prosthetic=True, difficulty=0, seed=None)

	observation_d = env.reset(project = False)

	observation = process_obs_dict(observation_d)
	obs_size = observation.shape[0]
	print(obs_size)

	print('Creating Agent')

	agent = ddpg_agent(
			obs_size,
			discount_factor=0.98
			# .99 = 100 steps = 4 second lookahead
			# .985 = somewhere in between.
			# .98 = 50 steps = 2 second lookahead
			# .96 = 25 steps = 1 second lookahead
			)

	'''
			processed_dims,
			e.action_space,
			discount_factor=.98,
			# .99 = 100 steps = 4 second lookahead
			# .985 = somewhere in between.
			# .98 = 50 steps = 2 second lookahead
			# .96 = 25 steps = 1 second lookahead
			stack_factor=1,
			train_multiplier=1
	'''

	

	noise_level = 0.55
	noise_decay_rate = 0.005
	noise_floor = 0.0001
	noiseless = 0.0001

	# Where does the noise process come in...

	#def play(noise_level):

		# create env

		# agent.play, will include skip factor
		# BE PREPARED FOR MEMORY LEAK HERE, kill and restart - make it automatic...



	def train(eps, save_interval = 200, frameskip = 4):

		global noise_level, env

		current_best_reward = 0

		

		for i in range(eps):

			noise_level *= (1 - noise_decay_rate)
			noise_level = max(noise_floor, noise_level)

			# nl = noise_level if np.random.uniform()>0.05 else noiseless
			nl = noise_level if i%20!=0 else noiseless
			# nl = noise_level
			# nl = noise_level * np.random.uniform() + 0.01

			print('ep',i+1,'/',eps,'noise_level',nl)

			# Change play function here...
			#playifavailable(nl)
			#play(nl)
			# no wait, this will recreate the env every time
			# maybe create the env once for the whole training session...
			# and then feed it into the agent.play
			ep_reward = agent.play_one_ep(env, nl, frameskip)

			if (i+1) % save_interval == 0:

				save()

			if ep_reward > current_best_reward:

				print('New best!')
				agent.save_weights('best_saves/')

				current_best_reward = ep_reward




	def save():

		agent.save_weights()
		agent.rpm.save('rpm.pickle')

		with open('reward_history.p', 'wb') as f:

			pickle.dump(agent.reward_collector, f)

	def load_weights():
		agent.load_weights()

	def load_memory():
		agent.rpm.load('rpm.pickle')

	def test(frameskip = 1):

		env = RunEnv(visualize=False)

		observation_d = env.reset(project = False)
		#observation = process_obs_dict(observation_d)


		total_reward = 0
		steps = 0

		while True:

			#a = AGENT OUTPUT
			observation = process_obs_dict(observation_d)
			a, q = agent.act(observation)

			for _ in range(frameskip):

				observation_d, reward, done, info = env.step(a, project = False)
				#observation = process_obs_dict(observation_d)

				total_reward += reward 
				steps += 1

			#print(observation)

			print(steps, 'total reward:', total_reward)
			
			if done:

				break

		print('finished testing!')

	def upload(frameskip = 1):

		from osim.http.client import Client

		apikey = open('apikey.txt').read().strip('\n')

		print('Using apikey:', apikey)

		remote_base = "http://grader.crowdai.org:1729"
		crowdai_token = apikey

		print('connecting...')
		client = Client(remote_base)

		observation_d = client.env_create(crowdai_token, env_id="ProstheticsEnv")
		#observation = process_obs_dict(observation_d)

		print('environment created! running...')

		#obs_collect = []
		#a_collect = []

		stepno= 0
		epino = 0
		total_reward = 0

		while True:

			#a = AGENT OUTPUT
			observation = process_obs_dict(observation_d)
			a, q = agent.act(observation)
			a = [float(i) for i in list(a)]

			#obs_collect.append(observation)
			#a_collect.append(a)

			for _ in range(frameskip):

				[observation_d, reward, done, info] = client.env_step(a, True)
				

				stepno += 1
				total_reward += reward

				print('step',stepno,'total reward',total_reward)

				if done:

					'''
					print('')
					print('saving...')
					print('')

					with open('upload_saves/upload_a_collect_' + str(epino) + '.p', 'wb') as f:

						pickle.dump(a_collect, f)

					with open('upload_saves/upload_obs_collect_' + str(epino) + '.p', 'wb') as f:

						pickle.dump(obs_collect, f)
					'''

					observation_d = client.env_reset()
					
	 
					print('>> episode',epino,' Done after',stepno,'got reward:',total_reward)
					print('')

					total_reward = 0
					stepno = 0
					epino += 1

					break

			if not observation_d:

				break

		print('Done! Submitting...')
		client.submit()



