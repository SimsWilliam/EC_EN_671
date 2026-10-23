#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%% Problem 8-3
print("Problem 8-3 ================================================================")

# Part (a) Make a plot of the data:
x = np.array([2, 2.5, 3, 5, 9])
y = np.array([-4.2, -5, 2, 1, 24.3])

plt.plot(x, y, ".")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

# Part (b) Make a plot of the data:

A = np.vstack([x, np.ones(len(x))]).T

coefficients = np.linalg.inv(A.T@A)@A.T@y

print(f"Coefficients for optimal linear regression: {coefficients}")

xs = np.array([2, 9])
ys = coefficients[0]*xs + coefficients[1]

plt.plot(x, y, ".", label="Data")
plt.plot(xs, ys, label="Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

# Part (c) Weight the first and last data points, as they are more trusted

W = np.diag([10, 1, 1, 1, 10])

weighted_coefficients = np.linalg.inv(A.T@W@A)@A.T@W@y

print(f"Coefficients for optimal weighted linear regression: {weighted_coefficients}")

weighted_ys = weighted_coefficients[0]*xs + weighted_coefficients[1]

plt.plot(x, y, ".", label="Data")
plt.plot(xs, weighted_ys, label="Weighted Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

plt.plot(x, y, ".", label="Data")
plt.plot(xs, ys, label="Fit")
plt.plot(xs, weighted_ys, label="Weighted Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()

# %% Orthonormalized Polynomials

# %%
#%%  (From hw4.py)

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

steps = 10001
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
# %%
# Part a)
# We want to solve Rc = p for c, then multiply A*c
# In this case, because we have an orthonormal basis, R is the identity matrix. p is the dotting of e^t with the various legendre polynomials

R = np.eye(5)

p = np.zeros([5, 1])

f = np.exp(t_legendre)

for i in range(p.size):
    p[i] = legendre_dt*np.dot(legendre[:, i], f)

c = p

f_legendre_approx = legendre@c

error = f.ravel() - f_legendre_approx.ravel()

error_norm = np.sqrt(legendre_dt*np.sum(error*error))

print(f"Norm of the error for Legendre approximation: {error_norm}")

incorrect_error_norm = np.linalg.norm(error)

print(f"Euclidian norm of the Legendre approximation: {incorrect_error_norm}")

plt.plot(t_legendre, f, label="e^t")
plt.plot(t_legendre, f_legendre_approx, linestyle="--", label="e^t approximation")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()

# Part b)
# Now we do the exact same thing with the Chebyshev polynomials

p = np.zeros([5, 1])

f = np.exp(t_chebyshev)

for i in range(p.size):
    p[i] = chebyshev_dt*np.sum(chebyshev[:, i]*f*c)

c = p

f_chebyshev_approx = chebyshev@c

error = f.ravel() - f_chebyshev_approx.ravel()


error_norm = np.sqrt(chebyshev_dt*np.sum(error*error*w))
error_norm = np.sqrt(chebyshev_dt*np.sum(error*error))
# error_norm = np.sqrt(np.trapezoid(error*error*w, t_chebyshev))

print(f"Norm of the error for Chebyshev approximation: {error_norm}")

incorrect_error_norm = np.linalg.norm(error)

print(f"Euclidian norm of the Chebyshev approximation: {incorrect_error_norm}")

plt.plot(t_chebyshev, f, label="e^t")
plt.plot(t_chebyshev, f_chebyshev_approx, linestyle="--", label="e^t approximation")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.legend()
plt.show()

# %% Image Processing

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the matrices needed for this problem
data = loadmat('prob2.mat')  # Load the .mat file
A = data['A']  # Assuming A is stored in the .mat file
x = data['x']  # Assuming x is also stored in the .mat file

# Display the 20 basis images in matrix A
fig, axs = plt.subplots(4, 5, figsize=(10, 10))
for kk in range(20):
    ax = axs[kk // 5, kk % 5]
    ax.imshow(A[:, :, kk], cmap='gray')
    ax.axis('off')
plt.show()

# Display the image x
plt.figure()
plt.imshow(x, cmap='gray')
plt.axis('off')
plt.show()

A_flat = A.reshape(-1, *A.shape[2:])
x_flat = x.reshape(-1)


# We can easily find the projection of x onto the basis images by xhat = A(A^TA)^-1A^Tx

x_proj_flat = A_flat@np.linalg.inv(A_flat.T@A_flat)@A_flat.T@x_flat

x_proj = x_proj_flat.reshape([256, 256])

plt.figure()
plt.imshow(x_proj, cmap='gray')
plt.axis('off')
plt.show()

x_rejection_flat = x_flat - x_proj_flat

x_rejection = x_rejection_flat.reshape([256, 256])

plt.figure()
plt.imshow(x_rejection, cmap='gray')
plt.axis('off')
plt.show()

# %%
