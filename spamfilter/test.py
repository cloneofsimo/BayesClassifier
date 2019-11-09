import numpy as np

x = np.arange(10)
np.save("hi", x)

y = np.load("hi.npy")

print(y[5])
