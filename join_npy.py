import numpy as np

p = np.load('patches.npy')
p1 = np.load('patches2.npy')
p2 = np.load('patches3.npy')

len_all = p.shape[0] + p1.shape[0] + p2.shape[0]
p_all = np.zeros((len_all, p.shape[1], p.shape[2]))

p_all[0:p.shape[0], :, :] = p
p_all[p.shape[0]:p.shape[0] + p1.shape[0], :, :] = p1
p_all[p.shape[0] + p1.shape[0]:len_all, :, :] = p2

np.save('p_all', p_all)