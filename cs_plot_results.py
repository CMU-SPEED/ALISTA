import numpy as np
import matplotlib.pyplot as plt

ista_data = np.load('./results/m256_n512_k0.0_p0.1_sinf/ista.npz')
ista_nmse = ista_data['nmse']
ista_x = ista_data['xestimate']

ls_data = np.load('./results/m256_n512_k0.0_p0.1_sinf/least_square.npz')
ls_nmse = ls_data['nmse']
ls_x = ls_data['xestimate']

lista_data = np.load('/Users/laixishi/Dropbox/research/projects/drapa-lista/ALISTA-master/results/m256_n512_k0.0_p0.1_sinf/LISTA_complex_T16_lam0.4_t_s_0.npz')
lista_nmse = lista_data['nmse']

lista_xs_ = lista_data['xhs_']

xtrue = lista_data['xtrue']

#-------- plot the nmse
plt.figure()

T = len(lista_xs_)
print(len(lista_xs_))
ista_num = 2000
# plt.plot(range(ista_num), ls_nmse * np.ones((ista_num,)), label='Least Square')
plt.plot(range(T), lista_nmse, label='LISTA')
# plt.plot(range(ista_num), ista_nmse[0:ista_num], label='ISTA')

plt.ylim(-50, 1)
plt.xlabel('Number of iterations')
plt.ylabel('NMSE')
plt.legend()
plt.show()

plt.figure()
ista_num = 5000
plt.plot(range(ista_num), ls_nmse * np.ones((ista_nmse.shape[0],)), label='Least Square')
plt.plot(range(ista_num), ista_nmse, label='ISTA')
plt.ylim(-50, 1)
plt.xlabel('Number of iterations')
plt.ylabel('NMSE')
plt.legend()
plt.show()


# ## plot the final recovery results
# real_data = np.load('/Users/laixishi/Dropbox/research/projects/drapa-lista/ALISTA-master/data/xtest_n512_p01.npz')
# xtrue = real_data['x']
# ## plot the final recovered inputs
# N = xtrue.shape[0]
# fig, axs = plt.subplots(3)
# markerline, stemlines, baseline = axs[0].stem(np.abs(xtrue[:, 0]), linefmt='r-.', markerfmt='ro', label='Ground truth')
# markerline, stemlines, baseline = axs[0].stem(np.abs(ls_x[:, 0]), linefmt='b-.', markerfmt='bo', label='Estimated by least square')
#
# axs[0].legend()
# axs[0].set_title('Results of least square')
#
# markerline, stemlines, baseline = axs[1].stem(np.abs(xtrue[:, 0]), linefmt='r-.', markerfmt='ro', label='Ground truth')
# markerline, stemlines, baseline = axs[1].stem(np.abs(ista_x[:, 0]), linefmt='g-.', markerfmt='go', label='Estimated by ISTA')
# axs[1].legend()
# axs[1].set_title('Results of ISTA')
#
# lista_x = np.sqrt(np.square(lista_xs_[16, 0:N, 0]) + np.square(lista_xs_[16, N:N*2, 0]))
# markerline, stemlines, baseline = axs[2].stem(np.abs(xtrue[:, 0]), linefmt='r-.', markerfmt='ro', label='Ground truth')
# markerline, stemlines, baseline = axs[2].stem(lista_x, linefmt='y-.', markerfmt='yo', label='Estimated by LISTA')
#
# axs[2].legend()
# axs[2].set_title('Results of LISTA')
# plt.show()