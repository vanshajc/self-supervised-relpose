import numpy as np

def eightpoint(pts1, pts2, K1, K2):
    if pts1.shape[1] == 2:
        pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=-1)
    if pts2.shape[1] == 2:
        pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=-1)

    pts1 = pts1 @ np.linalg.inv(K1).T
    pts2 = pts2 @ np.linalg.inv(K2).T

    A = []
    for i in range(len(pts1)):
        p1, p2 = pts1[i], pts2[i]
        A.append([p1[0]*p2[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1.0])
    
    u, s, v = np.linalg.svd(A)
    E =  v[-1].reshape((3,3))
    u,s,v = np.linalg.svd(E)
    s[-1] = 0
    s[:2] = s[:2].mean()
    E = u @ np.diag(s) @ v
    return E

def get_pose_from_E(E):
    U, s, Vt = np.linalg.svd(E)
    s[-1] = 0

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    u3 = U[:, 2, None]
    opts = [np.block([U @ W @ Vt, u3]), np.block([U @ W @ Vt, -u3]), np.block([U @ W.T @ Vt, u3]), np.block([U @ W.T @ Vt, -u3])]
    
    for i, opt in enumerate(opts):
        if np.linalg.det(opt[:3,:3]) < -0.9:
            opts[i] = -opt

    return opts