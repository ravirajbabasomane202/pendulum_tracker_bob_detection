import numpy as np

# Fix: Change (6, 1) to (6, 5) to match 5 columns
matrix = np.zeros((6, 5))

# Define the exact row and column coordinates for your numbers
rows = np.arange(1, 6)  # [1, 2, 3, 4, 5]
cols = np.arange(0, 5)  # [0, 1, 2, 3, 4]

# Define the values you want to place at those coordinates
values = np.arange(2, 7)  # [2., 3., 4., 5., 6.]

# Insert the values into the matrix
matrix[rows, cols] = values

print(matrix)
