import numpy as np
import sympy as sp

def find_SVD(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    AT = np.transpose(A)
    A_AT = np.matmul(A, AT)
    AT_A = np.matmul(AT, A)
    eigvals, eigvecs = np.linalg.eigh(AT_A)
    sorted_eigvals_indexes = sorted(range(len(eigvals)), key=lambda i: eigvals[i], reverse=True)
    sorted_eigvals = eigvals[sorted_eigvals_indexes]
    sorted_eigvecs = eigvecs[:, sorted_eigvals_indexes]
    # all eigenvalues of symmetric matrices like AT_A are positive so any negative value is an error that can be corrected with an abs
    sigmas = np.sqrt(np.abs(sorted_eigvals))
    V = sorted_eigvecs
    U = compute_U_matrix(A, V, sigmas)
    S = np.zeros(A.shape)
    np.fill_diagonal(S, sigmas)

    return (U, S, V)
    
def compute_U_matrix(A: np.ndarray, V: np.ndarray, sigmas: np.ndarray):
    assert(np.ndim(A) == 2)
    m = A.shape[0]
    n = A.shape[1]
    r = min(m, n)
    vecs = []
    for i in range(r):
        vecs.append(np.matmul(A, V[:, i]) / sigmas[i])
    U = np.column_stack(vecs)
    if (m > r):
        # Find the null space basis of A^T using orthogonalization
        identity = np.eye(m)
        null_space_basis = identity - U @ U.T  # Project orthogonal to U_partial
        Q_additional, _ = np.linalg.qr(null_space_basis)  # Orthonormalize
        
        # Take only the additional columns needed
        Q_additional = Q_additional[:, r:]
        
        # Combine U_partial with the additional orthonormal vectors
        U = np.hstack((U, Q_additional))
    return U
    
def vec_length(vector: np.ndarray):
    assert(vector.ndim == 1)
    return np.sqrt(np.sum(np.square(vector)))

def recombine_SVD(U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int):
    # (mxk) (kxk) (kxn)
    # return original
    if (k <= 0): return np.matmul(np.matmul(U, S), V.transpose())

    m = S.shape[0]
    n = S.shape[1]

    assert k <= max(m, n)
    reduced_U = U[:, :k]
    reduced_S = S[:k, :k]
    reduced_V = V[:, :k]
    return np.matmul(np.matmul(reduced_U, reduced_S), reduced_V.transpose())



if __name__ == "__main__":
    # matrix = np.array([
    #     [3,2,2],
    #     [2,3,-2]
    # ])
    # find_SVD(matrix)
    # matrix1 = np.array([
    #     [1,2,3,4],
    #     [5,6,7,8],
    #     [9,10,11,12],
    #     [13,14,15,16],
    #     [17,18,19,20]
    # ])
    # find_SVD(matrix1)
    matrix2 = np.array([
        [2,1],
        [1,-1]    
    ])
    U, S, V = find_SVD(matrix2)
    print(U)
    print(S)
    print(V)
    tU, tS, tVh = np.linalg.svd(matrix2)
    tV = tVh.transpose()
    print(tU)
    print(tS)
    print(tV)    
    
