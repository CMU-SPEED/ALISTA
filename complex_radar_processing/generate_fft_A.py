from scipy.linalg import dft
import numpy as np

def get_dft_A(M, N):
    F = dft(N)
    random_indices = np.random.choice(F.shape[0], size=M, replace=False)
    A = F[random_indices,:]
    return A

# # test
# B = get_dft_A(100,200)


