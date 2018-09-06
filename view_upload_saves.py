import pickle 

# Total reward at the end was -320.42, -320.32
# Stable over two runs of upload()

def unpick(fn):

	with open(fn, 'rb') as f:

		z = pickle.load(f)

	return z

a_collect = []
o_collect = []

for i in range(4):

	a = unpick('upload_saves/upload_a_collect_' + str(i) + '.p')
	o = unpick('upload_saves/upload_obs_collect_' + str(i) + '.p')

	a_collect.append(a)
	o_collect.append(o)

'''
View Actions

for i in range(2):

	for j in range(5):

		print(a_collect[i][j][:])

	print('')

'''

# View 

#obs.extend() # x, y, z
#obs.extend(observation['misc']['mass_center_vel']) # x, y, z
#obs.extend(observation['misc']['mass_center_acc']) # x, y, z

print(o_collect[0][0]['misc']['mass_center_pos'])
print(o_collect[1][0]['misc']['mass_center_pos'])
print(o_collect[2][0]['misc']['mass_center_pos'])
print(o_collect[3][0]['misc']['mass_center_pos'])
print('')

print(o_collect[0][0]['misc']['mass_center_vel'])
print(o_collect[1][0]['misc']['mass_center_vel'])
print(o_collect[2][0]['misc']['mass_center_vel'])
print(o_collect[3][0]['misc']['mass_center_vel'])
print('')

print(o_collect[0][0]['misc']['mass_center_acc'])
print(o_collect[1][0]['misc']['mass_center_acc'])
print(o_collect[2][0]['misc']['mass_center_acc'])
print(o_collect[3][0]['misc']['mass_center_acc'])
print('')

print(o_collect[0][0]['joint_pos']['hip_r'])
print(o_collect[1][0]['joint_pos']['hip_r'])
print(o_collect[2][0]['joint_pos']['hip_r'])
print(o_collect[3][0]['joint_pos']['hip_r'])
print('')

print(o_collect[0][0]['joint_pos']['knee_r'])
print(o_collect[1][0]['joint_pos']['knee_r'])
print(o_collect[2][0]['joint_pos']['knee_r'])
print(o_collect[3][0]['joint_pos']['knee_r'])
print('')

print(o_collect[0][0]['joint_pos']['ankle_r'])
print(o_collect[1][0]['joint_pos']['ankle_r'])
print(o_collect[2][0]['joint_pos']['ankle_r'])
print(o_collect[3][0]['joint_pos']['ankle_r'])
print('')

bs = ['body_pos', 'body_vel', 'body_acc']
parts = ['pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r']

for b in bs:

	for p in parts:

		print(b, p)

		for i in range(4):

			print(o_collect[i][0][b][p])

		print('')


