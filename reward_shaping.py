


def reward_shaping(obs_dict, input_reward):

	# Assuming that:
	# X is along the direction of movement
	# Y is up and down
	# Z is Left and right

	com_z = obs_dict['misc']['mass_center_pos'][2]

	head_x = obs_dict['body_pos']['head'][0]
	pelvis_x = obs_dict['body_pos']['pelvis'][0]

	head_z = obs_dict['body_pos']['head'][2]
	pelvis_z = obs_dict['body_pos']['pelvis'][2]

	# Not sure what the units are here...
	# Just using the penalty from the stanford guy
	l_knee_angle = obs_dict['joint_pos']['knee_l'][0]
	r_knee_angle = obs_dict['joint_pos']['knee_r'][0]


	lean_forward_penalty = 0.05 * min(0.3, max(0, pelvis_x - head_x - 0.3))
	dont_lean_sideways_penalty = 0 #1.0 * (head_z - pelvis_z) ** 2
	stay_centered_penalty = 0.1 * (pelvis_z ** 2)
	left_knee_bend_penalty = 0.50 * max(0, l_knee_angle)# - 0.1)
	right_knee_bend_penalty = 0.50 * max(0, r_knee_angle)# - 0.1)

	#print(lean_forward_penalty, dont_lean_sideways_penalty, stay_centered_penalty, left_knee_bend_penalty)

	penalty = lean_forward_penalty + dont_lean_sideways_penalty + stay_centered_penalty + left_knee_bend_penalty + right_knee_bend_penalty

	return input_reward + penalty

'''
def reward_shaping(obs_dict, input_reward):

	# Assuming that:
	# X is along the direction of movement
	# Y is up and down
	# Z is Left and right

	com_z = obs_dict['misc']['mass_center_pos'][2]

	head_x = obs_dict['body_pos']['head'][0]
	pelvis_x = obs_dict['body_pos']['pelvis'][0]

	head_z = obs_dict['body_pos']['head'][2]
	pelvis_z = obs_dict['body_pos']['pelvis'][2]

	# Not sure what the units are here...
	# Just using the penalty from the stanford guy
	l_knee_angle = obs_dict['joint_pos']['knee_l'][0]


	lean_forward_penalty = 1.0 * min(0.3, max(0, pelvis_x - head_x - 0.3))
	dont_lean_sideways_penalty = 1.0 * (head_z - pelvis_z) ** 2
	stay_centered_penalty = 1.0 * (pelvis_z ** 2)
	left_knee_bend_penalty = 0.30 * max(0, l_knee_angle - 0.1)

	#print(lean_forward_penalty, dont_lean_sideways_penalty, stay_centered_penalty, left_knee_bend_penalty)

	penalty = lean_forward_penalty + dont_lean_sideways_penalty + stay_centered_penalty + left_knee_bend_penalty

	return input_reward + penalty

'''