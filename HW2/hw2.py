import numpy as np
data = np.genfromtxt("EC_EN_671/HW2/Homework2_kNN_Data.csv", delimiter=",", names=True)
X = np.column_stack((data["feature_1"], data["feature_2"]))
labels = data["label"].astype(int)
q = np.array([[0.15],
              [0.01]])      # column vector
q_row = q.ravel()            # convenient shape for distances to rows of X

# Part 1 -------------------------------------------------------------
def ell_p_metric(q, data, p):
    if p != np.inf:
        distances = np.sum(np.abs(data - q)**p, axis=1)**(1/p)
    else:
        distances = np.max(np.abs(data - q), axis=1)

    return distances

# Part 2 -------------------------------------------------------------
distanceses = []

metrics = [1, 2, np.inf]

for i in metrics:
    distanceses.append(ell_p_metric(q_row, X, i))

distanceses = np.array(distanceses)

least_to_greatest = np.argsort(distanceses, axis=1)

k_nearest = least_to_greatest[:, :3]

labels[k_nearest]

k_nearest_labels = labels[k_nearest]

print(f"k nearest_labels: \n {k_nearest_labels}")

# Part 2 ---------------------------------------------------------
X[:, 1] *= 10
q_row[1] *= 10

distanceses = []

metrics = [1, 2, np.inf]

for i in metrics:
    distanceses.append(ell_p_metric(q_row, X, i))

distanceses = np.array(distanceses)

least_to_greatest = np.argsort(distanceses, axis=1)

k_nearest = least_to_greatest[:, :3]

labels[k_nearest]

k_nearest_labels = labels[k_nearest]

print(f"k nearest_labels: \n {k_nearest_labels}")