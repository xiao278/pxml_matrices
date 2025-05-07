import numpy as np
import time

def calculate_householder(mat: np.ndarray, iter: int):
    # iter 1 output is mxm
    # iter 2 output is (m-1)x(m-1)
    n = mat.shape[0]
    x = mat[iter:,iter]
    z = np.zeros(x.shape)
    z[0] = 1
    v = x + np.linalg.norm(x, 2) * z
    v = v.reshape(-1, 1)
    I = np.identity(n - iter)
    H = I - 2 * np.matmul(v, v.transpose()) / (np.linalg.norm(v) ** 2)
    H = np.pad(H, pad_width=((iter, 0), (iter, 0)), mode='constant', constant_values=0)
    for i in range(iter):
        H[i,i] = 1
    return H

def calculate_QR(A: np.ndarray):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    Q = np.identity(n)
    R = A
    for iter in range(0, n-1):
        H = calculate_householder(R, iter)
        Q = np.matmul(Q, H)
        R = np.matmul(H, R)
    # round_to_zero(Q)
    # round_to_zero(R)
    return Q, R

def round_to_zero(matrix: np.ndarray):
    '''rounds very small values (1e-14) to zero'''
    epsilon = 1e-14
    matrix[np.abs(matrix) < epsilon] = 0

def has_converged(matrix, tol=1e-10):
    below_diag = np.tril(matrix, k=-1)
    diag = np.diag(matrix)
    max_below_diag = np.max(np.abs(below_diag))
    # print(max_below_diag)
    return max_below_diag < tol

#try using different shift as it converges
#check for change in diagonal if they dont change much it could mean it converged

def calculate_eig(matrix: np.ndarray):
    A = matrix
    m = matrix.shape[0]
    n = matrix.shape[1]
    r = min(m,n)
    Q = np.eye(m,n)
    R = None
    iters = 0
    while (not has_converged(A) and iters < 1000):
        alpha = A[r-1,r-1]
        shift_matrix = np.eye(m, n) * alpha
        A_shift = A - shift_matrix
        Q, R = calculate_QR(A_shift)
        A_new = R @ Q + shift_matrix
        # print(np.diag(A) - np.diag(A_new))
        A = A_new
        iters += 1
    print(iters)
    return Q, np.diag(A)


if __name__ == "__main__":
    matrix2 = np.array([
        [1,2,3,4],
        [5,6,70,8],
        [9,101,11,12],
        [13,-5,15,16]    
    ])
    matrix_10x10 = np.array([
        [ 0.09762701,  0.43037873,  0.20552675,  0.08976637, -0.1526904,  
        0.29178823, -0.12482558,  0.783546,    0.92732552, -0.23311696],
        [ 0.58345008,  0.05778984,  0.13608912,  0.85119328, -0.85792788, 
        -0.8257414,  -0.95956321,  0.66523969,  0.5563135,   0.7400243 ],
        [ 0.95723668,  0.59831713, -0.07704128,  0.56105835, -0.76345115,  
        0.27984204, -0.71329343,  0.88933783,  0.04369664, -0.17067612],
        [-0.47088878,  0.54846738, -0.08769934,  0.1368679,  -0.9624204,  
        0.23527099,  0.22419145,  0.23386799,  0.88749616,  0.3636406 ],
        [-0.2809842,  -0.12593609,  0.39526239, -0.87954906,  0.33353343,  
        0.34127574, -0.57923488, -0.7421474,  -0.3691433,  -0.27257846],
        [ 0.14039354, -0.12279697,  0.97674768, -0.79591038, -0.58224649,  
        -0.67738096,  0.30621665, -0.49341679, -0.06737845, -0.51114882],
        [-0.68206083, -0.77924972,  0.31265918, -0.7236341,  -0.60683528,  
        -0.26254966,  0.64198646, -0.80579745,  0.67588981, -0.80780318],
        [ 0.95291893, -0.0626976,   0.95352218,  0.20969104,  0.47852716,  
        -0.92162442, -0.43438607, -0.75960688, -0.4077196,  -0.76254456],
        [-0.36403364, -0.17147401, -0.87170501,  0.38494424,  0.13320291,  
        -0.46922102,  0.04649611, -0.81211898,  0.15189299,  0.8585924 ],
        [-0.3628621,   0.33482076, -0.73640428,  0.43265441, -0.42118781,  
        -0.63361728,  0.17302587, -0.95978491,  0.65788006, -0.99060905]
    ])
    matrix_10x10 = matrix_10x10 @ matrix_10x10.transpose()
    matrix3 = np.array([
        [0,-1],
        [1,0]
    ])

    start = time.time()
    Q, eigvals = calculate_eig(matrix_10x10)
    end = time.time()
    print(end - start)
    eigvals = sorted(eigvals)
    print(eigvals)
    start1 = time.time()
    result = np.linalg.eigvals(matrix_10x10)
    end1 = time.time()
    print(end1 - start1)
    result = sorted(result)
    print(result)
    print(np.subtract(eigvals, result))

    