#%%
import numpy as np
import scipy as sp
from scipy.linalg import null_space
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #%% Problem 2.13-67
    print("Problem 2.13-67 =============")

    p_1 = np.array([1, 2, 3, 4]).reshape([4, 1])
    p_2 = np.array([4, -2, -6, -7]).reshape([4, 1])
    p_3 = np.array([3, 4, -2, 1]).reshape([4, 1])

    A = np.hstack([p_1, p_2, p_3])

    x = np.array([1, 2, 3, 7]).reshape([4, 1])

    x_hat = A@np.linalg.inv(A.T@A)@A.T@x

    print(f"x_hat: \n{x_hat}")

    e = x - x_hat

    # print(np.dot(p_1.ravel(), e.ravel()), np.dot(p_2.ravel(), e.ravel()), np.dot(p_3.ravel(), e.ravel()))

    B = x.T@A

    C = null_space(B)

    print(f"C: {C}")

    x_perp_in_V = A@C

    print(f"AC: \n{x_perp_in_V}")

    #%% Problem 6. Computational Problem: Legendre and Chebyshev Polynomials

    #%%
    # Part a)
    steps = 10001
    dt = 2/(steps-1)
    legendre_dt = dt

    t = np.linspace(-1, 1, steps)
    t_legendre = t

    p = np.zeros([steps, 5])

    for i in range(5):
        p[:, i] = t**i

    q = np.zeros([steps, 5])

    for i in range(5):
        e = p[:, i].copy()
        for j in range(i):
            e -= dt*np.dot(p[:, i], q[:, j])*q[:, j]
        q[:, i] = e / np.sqrt(dt*np.dot(e, e))

    legendre = q
    plt.plot(t, q)
    plt.xlabel("t")
    plt.ylabel("p_i(t)")
    plt.legend(["p_0", "p_1", "p_2", "p_3", "p_4"])
    plt.show()

    q_classic = q / q[-1, :] # This is not part of Gram-Schmidt, but it is part of the traditional definition of the Legendre scaling

    plt.plot(t, q_classic)
    plt.xlabel("t")
    plt.ylabel("p_i(t)")
    plt.legend(["p_0", "p_1", "p_2", "p_3", "p_4"])
    plt.show()

    #%% 
    # Part b)
    steps = 50001
    dt = 2/(steps-1)
    chebyshev_dt = dt

    t = np.linspace(-1, 1, steps)
    t_chebyshev = t

    p = np.zeros([steps, 5])

    t_safe = t.copy()
    t_safe[0] = 0
    t_safe[-1] = 0
    w = 1 / np.sqrt(1 - t_safe**2)
    w[0] = 0
    w[-1] = 0
    print(w[0], w[1], w[-2], w[-1])
    for i in range(5):
        p[:, i] = t**i

    q = np.zeros([steps, 5])

    for i in range(5):
        e = p[:, i].copy()
        for j in range(i):
            e -= dt*np.sum(p[:, i]*q[:, j]*w)*q[:, j]
        q[:, i] = e / np.sqrt(dt*np.sum(e*e*w))

    chebyshev = q
    plt.plot(t[1:-1], q[1:-1, :])
    plt.xlabel("t")
    plt.ylabel("p_i(t)")
    plt.legend(["p_0", "p_1", "p_2", "p_3", "p_4"])
    plt.show()

    q_classic = q / q[-1, :] # This is not part of Gram-Schmidt, but it is part of the traditional definition of the Chebyshev scaling

    plt.plot(t, q_classic)
    plt.xlabel("t")
    plt.ylabel("p_i(t)")
    plt.legend(["p_0", "p_1", "p_2", "p_3", "p_4"])
    plt.show()
    # %%
