import numpy as np

# np.sort()
np1 = np.array([6,7,9,0,2,4])
print(np1)
print(np.sort(np1))

# Alphabetical
np2 = np.array(["Stefano", "Dave", "Sabrina", "Karina"])
print(np2)
print(np.sort(np2))

# Booleans
np3 = np.array([True, False, False, True])
print(np3)
print(np.sort(np3))

# Return a copy not change the original
print(np1)
print(np.sort(np1))
print(np1)

# 2-D
np4 = np.array([[6,7,9,4],[3,2,8,5]])
print(np4)
print(np.sort(np4))