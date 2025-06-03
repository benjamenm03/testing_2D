import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the matrices from the CSV files
matrix1 = np.genfromtxt('temperature_grid.csv', delimiter=',')
matrix2 = np.genfromtxt('interpolated_grid.csv', delimiter=',')


# Create a grid of indices for both matrices
x1 = np.arange(matrix1.shape[1])
y1 = np.arange(matrix1.shape[0])
X1, Y1 = np.meshgrid(x1, y1)

x2 = np.arange(matrix2.shape[1])
y2 = np.arange(matrix2.shape[0])
X2, Y2 = np.meshgrid(x2, y2)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the first matrix in 2D
c1 = axs[0, 0].pcolormesh(X1, Y1, matrix1, cmap='viridis')
fig.colorbar(c1, ax=axs[0, 0])
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_title('Temperature Grid')

# Plot the second matrix in 2D
c2 = axs[0, 1].pcolormesh(X2, Y2, matrix2, cmap='viridis')
fig.colorbar(c2, ax=axs[0, 1])
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].set_title('Interpolated Temperature Grid')

# Plot the first matrix in 3D
ax3d1 = fig.add_subplot(223, projection='3d')
ax3d1.plot_surface(X1, Y1, matrix1, cmap='viridis')
ax3d1.set_xlabel('X')
ax3d1.set_ylabel('Y')
ax3d1.set_zlabel('Z')
ax3d1.set_title('Temperature Grid in 3D')

# Plot the second matrix in 3D
ax3d2 = fig.add_subplot(224, projection='3d')
ax3d2.plot_surface(X2, Y2, matrix2, cmap='viridis')
ax3d2.set_xlabel('X')
ax3d2.set_ylabel('Y')
ax3d2.set_zlabel('Z')
ax3d2.set_title('Interpolated Temperature Grid in 3D')

# Adjust layout
plt.tight_layout()
plt.show()
