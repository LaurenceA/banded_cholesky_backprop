import time
import numpy as np
from scipy.linalg import cholesky_banded
import torch
from torch.autograd import gradcheck, Function
from numba import jit

class GBTRF(Function):
    @staticmethod
    def forward(ctx, A):
        U = torch.tensor(cholesky_banded(A.data.numpy()))
        for i in range(M-1):
            for j in range(M-1-i):
                U[i, j] = 0
        ctx.save_for_backward(A, U)
        return U

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, dU):
        A, U = ctx.saved_tensors
        A = A.data.numpy()
        U = U.data.numpy()
        dU = dU.data.numpy()

        M, N = A.shape

        dA  = np.zeros_like(A)
        dUp = np.copy(dU)

        #Inner modifies dA
        inner(A, U, dA, dU, dUp)

        return torch.tensor(dA), 

@jit(nopython=True)
def inner(A, U, dA, dU, dUp):
    M, N = A.shape
    for i in range(N-1, -1, -1):
        #### Diagonal

        ## Set diagonal of A
        dA[-1, i] += dUp[-1, i] / U[-1, i] / 2

        ## Push from diagonal to off-diagonal
        for od in range(1, M):
            j = M-od-1
            dUp[j, i] -= dUp[-1, i] * U[j, i] / U[-1, i]

        #### Off-diagonal
        for od in range(1, min(i+1, M)):
            j = M-od-1

            ## Set off-diagonal of A
            dA[j, i] += dUp[j, i] / U[-1, i-od]
            ## Push from off-diagonal to diagonal
            dUp[-1, i-od] -= dUp[j, i] * U[j, i] / U[-1, i-od]

            ## Push from off-diagonal to off-diagonal
            for odp in range(od+1, M):
                #Product over two terms: U[j, i], U[jp, i], U[jp+od, i-od]
                jp = M-odp-1
                dUp[jp+od, i-od] -= dUp[j, i] * U[jp, i]       / U[-1, i-od]
                dUp[jp, i]       -= dUp[j, i] * U[jp+od, i-od] / U[-1, i-od]

gbtrf = GBTRF.apply

if __name__== "__main__":
    N = 6
    M = 4
    L = np.random.rand(N,N)
    L = np.tril(L)
    L = np.triu(L, -(M-1))
    mat = L @ L.T

    banded_mat = np.zeros([M, N])
    for i in range(M):
        banded_mat[(M-1-i), i:] = mat[range(N-i), range(i,N)]

    gradcheck(gbtrf, (torch.tensor(banded_mat, requires_grad=True), ), eps=1e-6, atol=1e-4)


    N = 10000000
    M = 4

    A = np.zeros([M, N])
    for od in range(M):
        A[M-od-1, :] = 2**(-od)

    U = cholesky_banded(A)
    dA = np.zeros_like(A)
    dU = np.random.randn(*A.shape)
    dUp = np.copy(dU)

    inner(A, U, dA, dU, dUp)

    start = time.time()
    #U = cholesky_banded(A)
    inner(A, U, dA, dU, dUp)
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))
