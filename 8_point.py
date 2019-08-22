import numpy as np


def construct_matrix_A(points_1_normalized, points_2_normalized):
    A = np.zeros((8,9))

    for i in range(8):  
        A[i] = [
            points_1_normalized[i, 0] * points_2_normalized[i, 0], points_1_normalized[i ,0] * points_2_normalized[i,1], points_1_normalized[i,0],
            points_1_normalized[i, 1] * points_2_normalized[i, 0], points_1_normalized[i, 1] * points_2_normalized[i,1], points_1_normalized[i,1],
            points_2_normalized[i, 0], points_2_normalized[i, 1], 1
        ]

    return A

def alg_8_point(points_1, points_2):
    """
    Steps
        normalize
        find matrix A
        SVD of ATA
        find matrix F
        Enforce rank 2 constraint 
        Unnormalize
    """
    points_1 = np.hstack((points_1, np.ones((points_1.shape[0], 1))))
    points_2 = np.hstack((points_2, np.ones((points_2.shape[0], 1))))

    mean_1 = np.mean(points_1, axis=0)
    std_1 = np.std(points_2, axis=0)

    T1 = np.array([
        [1/std_1[0], 0, -mean_1[0] / std_1[0]],
        [0, 1/std_1[1], -mean_1[1] / std_1[1]],
        [0, 0, 1]
    ])

    mean_2 = np.mean(points_2, axis=0)
    std_2 = np.std(points_2, axis=0)
    T2 = np.array([
        [1 / std_2[0], 0, -mean_2[0] / std_2[0]],
        [0, 1 / std_2[1], -mean_2[1] / std_2[1]],
        [0, 0, 1]
    ])

    points_1_normalized = T1.dot(points_1.T).T
    points_2_normalized = T2.dot(points_2.T).T


    A = construct_matrix_A(points_1_normalized, points_2_normalized)

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    F = F / np.linalg.norm(F)

    U, S, V = np.linalg.svd(F)
    S[2] = 0 # make zero last singular val
    F2 = U.dot(np.diag(S).dot(V))

    F = np.dot(T1.T, np.dot(F2,T2))
    
    return F


if __name__ == '__main__':
    points_1 = np.random.randint(20, size=(8, 2))
    points_2 = np.random.randint(20, size=(8, 2))


    F = alg_8_point(points_1, points_2)
    points_1 = np.hstack((points_1, np.ones((points_1.shape[0], 1))))
    points_2 = np.hstack((points_2, np.ones((points_2.shape[0], 1))))

    total_error = 0
    for i in range(len(points_1)):
        print('Val close to 0: \n')
        print(points_2[i].dot(F.dot(points_1[i])))
        