import numpy as np
import matplotlib.pyplot as plt

t = np.array([5, 10, 20, 40, 80])
s = np.array([0.752, 0.661, 0.516, 0.401, 0.195])

A = np.hstack([t.reshape([-1, 1]), np.ones([len(t), 1])])

R = A.T@A 
p = A.T@np.log(s.reshape([-1, 1]))

c = np.linalg.inv(R)@p

M_0 = np.exp(c[1])
T_2 = -1/c[0]

print(M_0)
print(T_2)

ts_plot = np.linspace(5, 80, 100)
s_hat = M_0*np.exp(-ts_plot/T_2)

plt.plot(ts_plot, s_hat, label="Regression")
plt.plot(t, s, "*", label="data")

plt.xlabel("Time (ms)")
plt.ylabel("Signal")

plt.legend()
plt.show()