import numpy as np

def rel(listA, listB):

	return [listA[i] - listB[i] for i in range(len(listA))]

def process_obs_dict(obs_dict):

	p_p = obs_dict['body_pos']['pelvis']
	p_v = obs_dict['body_vel']['pelvis']
	p_a = obs_dict['body_acc']['pelvis']

	obs = []

	obs.extend(rel(obs_dict['misc']['mass_center_pos'], p_p)) # x, y, z
	obs.extend(rel(obs_dict['misc']['mass_center_vel'], p_v)) # x, y, z
	obs.extend(rel(obs_dict['misc']['mass_center_acc'], p_a)) # x, y, z

	# Absolute Joint Positions
	obs.extend(obs_dict['joint_pos']['ground_pelvis'])

	obs.extend(obs_dict['joint_pos']['hip_r'])
	obs.extend(obs_dict['joint_pos']['knee_r'])
	obs.extend(obs_dict['joint_pos']['ankle_r'])

	obs.extend(obs_dict['joint_pos']['hip_l'])
	obs.extend(obs_dict['joint_pos']['knee_l'])
	obs.extend(obs_dict['joint_pos']['ankle_l'])

	obs.extend(obs_dict['joint_vel']['ground_pelvis'])

	obs.extend(obs_dict['joint_vel']['hip_r'])
	obs.extend(obs_dict['joint_vel']['knee_r'])
	obs.extend(obs_dict['joint_vel']['ankle_r'])

	obs.extend(obs_dict['joint_vel']['hip_l'])
	obs.extend(obs_dict['joint_vel']['knee_l'])
	obs.extend(obs_dict['joint_vel']['ankle_l'])

	# Absolute Joint Acc

	obs.extend(obs_dict['joint_acc']['ground_pelvis'])

	obs.extend(obs_dict['joint_acc']['hip_r'])
	obs.extend(obs_dict['joint_acc']['knee_r'])
	obs.extend(obs_dict['joint_acc']['ankle_r'])

	obs.extend(obs_dict['joint_acc']['hip_l'])
	obs.extend(obs_dict['joint_acc']['knee_l'])
	obs.extend(obs_dict['joint_acc']['ankle_l'])

	b = ['body_pos', 'body_vel', 'body_acc']
	parts = ['femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
	rel_pel = [p_p, p_v, p_a]

	for i in b:

		for j in parts:

			obs.extend(rel(obs_dict[i][j], rel_pel[b.index(i)]))

	#obs_dict.append(obs_dict['body_pos']['pelvis'][0]) #x
	obs.append(obs_dict['body_pos']['pelvis'][1]) #y
	obs.append(obs_dict['body_pos']['pelvis'][2]) #z

	return np.array(obs)




'''
def process_obs_dict(obs_dict):

	obs = []

	obs.extend(obs_dict['misc']['mass_center_pos']) # x, y, z
	obs.extend(obs_dict['misc']['mass_center_vel']) # x, y, z
	obs.extend(obs_dict['misc']['mass_center_acc']) # x, y, z

	# Absolute Joint Positions
	obs.extend(obs_dict['joint_pos']['ground_pelvis'])

	obs.extend(obs_dict['joint_pos']['hip_r'])
	obs.extend(obs_dict['joint_pos']['knee_r'])
	obs.extend(obs_dict['joint_pos']['ankle_r'])

	obs.extend(obs_dict['joint_pos']['hip_l'])
	obs.extend(obs_dict['joint_pos']['knee_l'])
	obs.extend(obs_dict['joint_pos']['ankle_l'])

	obs.extend(obs_dict['joint_vel']['ground_pelvis'])

	obs.extend(obs_dict['joint_vel']['hip_r'])
	obs.extend(obs_dict['joint_vel']['knee_r'])
	obs.extend(obs_dict['joint_vel']['ankle_r'])

	obs.extend(obs_dict['joint_vel']['hip_l'])
	obs.extend(obs_dict['joint_vel']['knee_l'])
	obs.extend(obs_dict['joint_vel']['ankle_l'])

	# Absolute Joint Acc

	obs.extend(obs_dict['joint_acc']['ground_pelvis'])

	obs.extend(obs_dict['joint_acc']['hip_r'])
	obs.extend(obs_dict['joint_acc']['knee_r'])
	obs.extend(obs_dict['joint_acc']['ankle_r'])

	obs.extend(obs_dict['joint_acc']['hip_l'])
	obs.extend(obs_dict['joint_acc']['knee_l'])
	obs.extend(obs_dict['joint_acc']['ankle_l'])

	b = ['body_pos', 'body_vel', 'body_acc']
	parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']

	for i in b:

		for j in parts:

			obs.extend(obs_dict[i][j])











	return np.array(obs)

'''