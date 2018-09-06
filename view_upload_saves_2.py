import pickle 

def unpick(fn):

	with open(fn, 'rb') as f:

		z = pickle.load(f)

	return z

o = unpick('upload_saves/upload2_obs_collect_0.p')


def sample_obs(obs):

	print(obs['misc']['mass_center_pos'])
	print(obs['misc']['mass_center_vel'])
	print(obs['misc']['mass_center_acc'])

	print(obs['joint_pos']['hip_r'])
	print(obs['joint_pos']['knee_r'])
	print(obs['joint_pos']['ankle_r'])

	bs = ['body_pos', 'body_vel', 'body_acc']
	parts = ['pelvis', 'femur_r', 'pros_tibia_r']

	for b in bs:

		for p in parts:

			print(obs[b][p])

	print('')

for i in range(len(o)):

	print(i)
	sample_obs(o[i])