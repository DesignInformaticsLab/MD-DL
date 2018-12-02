import numpy as np

data_in = np.loadtxt('in')
data_out = np.loadtxt('out')
num_data = 1000
num_molecure = 100

# position is already bounded in (0,1)
pos_in = []
for i in range(num_molecure):
	pos_in += [data_in[:,7*i:7*i+3].reshape(num_data, 1,3)]
pos_in = np.concatenate(pos_in,1)
print(pos_in.shape)


force_in = []
for i in range(num_molecure):
	force_in += [data_in[:,7*i+3:7*i+6].reshape(num_data, 1,3)]
force_in = np.concatenate(force_in,1)
for i in range(3):
    force_in[:,:,i] = (force_in[:,:,i]-np.min(force_in[:,:,i])) / (np.max(force_in[:,:,i])-np.min(force_in[:,:,i]))
print(force_in.shape)

potential_in = data_in[:,-1]
print(potential_in.shape)





# position is already bounded in (0,1)
pos_out = []
for i in range(num_molecure):
	pos_out += [data_out[:,7*i:7*i+3].reshape(num_data, 1,3)]
pos_out = np.concatenate(pos_out,1)
print(pos_out.shape)


force_out = []
for i in range(num_molecure):
	force_out += [data_out[:,7*i+3:7*i+6].reshape(num_data, 1,3)]
force_out = np.concatenate(force_out,1)
for i in range(3):
    force_out[:,:,i] = (force_out[:,:,i]-np.min(force_out[:,:,i])) / (np.max(force_out[:,:,i])-np.min(force_out[:,:,i]))
print(force_out.shape)

potential_out = data_in[:,-2]
print(potential_out.shape)

num_itr = data_out[:,-1]

idx = np.arange(0,num_data,1)
idx_train = np.asarray([idx[i] for i in range(num_data) if i%5])
idx_test = np.asarray([idx[i] for i in range(num_data) if i%5==0])
training_data = {'pos_in':pos_in[idx_train],
                 'pos_out':pos_out[idx_train],
                 'force_in':force_in[idx_train],
                 'force_out':force_out[idx_train],
                 'potential_in':potential_in[idx_train],
                 'potential_out':potential_out[idx_train],
                 'num_itr':num_itr[idx_train]}
np.save('training_data', training_data)
testing_data = {'pos_in':pos_in[idx_train],
                 'pos_out':pos_out[idx_train],
                 'force_in':force_in[idx_train],
                 'force_out':force_out[idx_train],
                 'potential_in':potential_in[idx_train],
                 'potential_out':potential_out[idx_train],
                 'num_itr':num_itr[idx_train]}
np.save('testing_data', testing_data)

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.distplot(pos_in.flatten())
# plt.show()
# sns.distplot(pos_out.flatten())
# plt.show()
# sns.distplot(force_in.flatten())
# plt.show()
# sns.distplot(force_out.flatten())
# plt.show()
# sns.distplot(potential_in.flatten())
# plt.show()
# sns.distplot(potential_out.flatten())
# plt.show()
# sns.distplot(num_itr)
# plt.show()
#
#
