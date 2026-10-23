import numpy as np
import matplotlib.pyplot as plt

def inner(x, y, dt):
    return dt*np.sum(x*y)

def norm(x, dt):
    return np.sqrt(inner(x, x, dt))


# Part a)

substeps = 100

dt = 0.25 / substeps
p1 = np.array([1, 1, 1, 1], dtype=float)
p2 = np.array([1, 1, -1, 1], dtype=float)
p3 = np.array([1, 1, -1, -1], dtype=float)

p = np.vstack([p1, p2, p3]).T
p = np.repeat(p, substeps, axis=0)
q = np.zeros_like(p)

n = p.shape[1]

for i in range(n):
    p_i = p[:, i].copy()
    e = p_i
    for j in range(i):
        e -= inner(p_i, q[:, j], dt)*q[:, j]

    e_norm = norm(e, dt)

    q[:, i] = e / e_norm

q_norm_check = []

for i in range(n):
    for j in range(n):
        q_norm_check.append(inner(q[:, i], q[:, j], dt))

q_norm_check = np.array([q_norm_check])


# Part b)

big_ts = [0, 0.25, 0.5, 0.75, 1.0]
ts_plot = []
for i in range(4):
    ts_plot.append(np.linspace(big_ts[i], big_ts[i+1], substeps+1))
ts_plot = np.hstack(ts_plot)

insertions = np.array([0, 1, 2, 3])*substeps
q_plot = np.insert(q, insertions, q[insertions], axis=0)

print(f"Orthonormal Basis: \n {q[insertions]}")
print(f"Orhtonormal Basis Norms: {q_norm_check}")

plt.plot(ts_plot, q_plot)
plt.legend(["q_1", "q_2", "q_3"])
plt.show()

# Part c)

ts = np.linspace(0, 1, 4*substeps+1)
ts = ts[1:]

x = 2*ts

cross_correlation = np.zeros(n)

for i in range(n):
    cross_correlation[i] = inner(x, q[:, i], dt)

coeff = cross_correlation

x_hat = q@coeff

x_hat_plot = np.insert(x_hat, insertions, x_hat[insertions])


print(f"Fitted coefficients: {coeff}")

# plt.plot(ts, x_hat, label="Approximation")
plt.plot(ts_plot, x_hat_plot, label="Approximation")
plt.plot(ts, x, label="Truth")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()
