from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]] 
# create data set from 3 assumed center
cov = [[1, 0], [0, 1]] 
# ma tran chinh phuong
N = 500 
# Number of data
X0 = np.random.multivariate_normal(means[0], cov, N)
 # cov dung de xac dinh hinh dang cua cac diem du lieu
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0) 
# concatenate : connect 3 matrix at the end of the rows
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T 
# np.asarray : convert input to an vertical array 3N ( ex : N=2, 0 0 1 1 2 2 )

def kmeans_display(X, label):
    K = np.amax(label) + 1
    # Return the maximum of an array or maximum along an axis.
    X0 = X[label == 0, :] 
    # tach ma tran X theo chieu ngang thanh 3 ma tran X0, X1, X2 như ban dau
    X1 = X[label == 1, :] 
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8) 
    # X0[:, 0] : every row in columns 0
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8) 
    # X0, X1, X2 is a Nx2 array
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)
def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)] 
    # replace = False : khong bi trung lap gia tri 
    # x.shape[0] : number of rows, x.shape[1] : number of columns

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    # if you have 3 center it will return a array with 3 column show distance from data to center
    return np.argmin(D, axis = 1)
    # Returns the indices of the minimum values along an axis. show the cluster this data belong to


def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
    # slimilar to list type

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))  
        # center cuoi cung cua array center
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)

(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])
kmeans_display(X, labels[-1])
