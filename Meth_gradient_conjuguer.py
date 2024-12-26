import numpy as np
import matplotlib.pyplot as plt
import sys

X = np.array([0,1])
A = np.array([[3,5],[5,6]])

A.X = A@X

print(A.X)
sys.exit()
