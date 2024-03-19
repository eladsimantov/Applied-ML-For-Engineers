import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
r_initial, r_final = 1, 10  # Example values
theta_initial, theta_final = 0, 2*np.pi  # Full circle
t_initial, t_final = 0, 10  # Example values
Nr, Ntheta, Nt = 50, 50, 100  # Example grid sizes

# Function to compute xi(r, theta, t)
def xi_function(r, theta, t):
    # Replace this with your actual function
    return np.sin(r) * np.cos(theta) * np.sin(t)

# Create a meshgrid for r and theta
r = np.linspace(r_initial, r_final, Nr)
theta = np.linspace(theta_initial, theta_final, Ntheta)
r, theta = np.meshgrid(r, theta)

# Convert to Cartesian coordinates for plotting
x = r * np.cos(theta)
y = r * np.sin(theta)

# Prepare plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize animation frame
def init():
    ax.clear()
    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([-1, 1])  # Adjust based on your function's range

# Update function for animation
def update(frame):
    t = np.linspace(t_initial, t_final, Nt)[frame]
    z = xi_function(r, theta, t)
    ax.clear()
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_zlim([-1, 1])  # Adjust based on your function's range
    return fig

# Create animation
ani = FuncAnimation(fig, update, frames=Nt, init_func=init, blit=False)
plt.show()

# Save animation
# ani.save( ... )
