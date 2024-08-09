import matplotlib.pyplot as plt

def lorenz(x, y, z):
    dx = s * (y - x)
    dy = (r - z) * x - y
    dz = x * y - b * z
    
    x_new = x + dt * dx
    y_new = y + dt * dy
    z_new = z + dt * dz

    return x_new, y_new, z_new

b = 8.0 / 3.0
r = 28.0
s = 10.0
dt = 1e-3

def integrate_lorenz_system(x0, y0, z0):
    trajectory = [(x0, y0, z0)]
    
    for _ in range(num_steps):
        x, y, z = trajectory[-1]
        x_new, y_new, z_new = lorenz(x, y, z)
        trajectory.append((x_new, y_new, z_new))
    return trajectory

num_steps = int(40 / dt)
x0, y0, z0 = -8.0, -1.0, 33.0

trajectory = integrate_lorenz_system(x0, y0, z0)

x_values = [point[0] for point in trajectory]
y_values = [point[1] for point in trajectory]
z_values = [point[2] for point in trajectory]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, lw=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz System Trajectory")

plt.show()

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 5(a) x vs z.
axs[0].plot(x_values, z_values, label="x vx z", color="red")
axs[0].set_xlabel('X')
axs[0].set_ylabel('Z')
axs[0].set_title('2-D Section: x vs. z')
axs[0].legend()

# 5(b) y vs z.
axs[1].plot(y_values, z_values, label="y vs z", color="green")
axs[1].set_xlabel('Y')
axs[1].set_ylabel('Z')
axs[1].set_title('2-D Section: y vs. z')
axs[1].legend()

# 5(c) x + y vs x - y
axs[2].plot([x + y for x, y in zip(x_values, y_values)], [x - y for x, y in zip(x_values, y_values)], label="x + y vs x - y")
axs[2].set_xlabel('X + Y')
axs[2].set_ylabel('X - Y')
axs[2].set_title('2-D Section: x + y vs. x - y')
axs[2].legend()

plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint

def lorenz_state(state, t):
    x, y, z = state
    dx = s * (y- x)
    dy = (r - z) * x - y
    dz = x * y - b * z
    return [dx, dy, dz]

t = np.arange(0.0, 40.0 , 0.001)

initial_state = [-8.0, -1.0, 33.0]
solution = odeint(lorenz_state, initial_state, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(solution[:,0], solution[:, 1], solution[:, 2], lw = 0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor")
plt.show()