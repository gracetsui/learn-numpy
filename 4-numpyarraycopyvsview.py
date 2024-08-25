import numpy as np

# Copy vs View
np1 = np.array([0,1,2,3,4,5])

# Create a view
np2 = np1.view()

print(f"Original NP1 {np1}")
print(f"Original NP2 {np2}")

np1[0] = 41

print(f"Changed np1 {np1}")
print(f"Changed np2 {np2}")

# Create a copy
np3 = np1.copy()
print(f"Original NP1 {np1}")
print(f"Original NP3 {np3}")

np1[0] = 41

print(f"Changed np1 {np1}")
print(f"Changed np3 {np3}")

