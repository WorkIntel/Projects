import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

"""
def get_accelerometer_data():
    # Replace with your real-time data acquisition logic
    x_acc = np.random.rand()
    y_acc = np.random.rand()
    z_acc = np.random.rand()
    return x_acc, y_acc, z_acc

# Create 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize plot variables
x_data = []
y_data = []
z_data = []
line = ax.plot(x_data, y_data, z_data, color='blue', linewidth=2)[0]

# Real-time plotting loop
while True:
    x_acc, y_acc, z_acc = get_accelerometer_data()
    x_data.append(x_acc)
    y_data.append(y_acc)
    z_data.append(z_acc)

    if len(x_data) > 30:
        x = x_data.pop(0)
        y = y_data.pop(0)
        z = z_data.pop(0)

    line.set_data(x_data, y_data)
    line.set_3d_properties(z_data)

    plt.draw()
    plt.pause(0.01)  # Adjust pause time for real-time update rate

"""

"""
    
#%%

def get_accelerometer_data():
    # Replace with your real-time data acquisition logic
    x_acc = np.random.rand()
    y_acc = np.random.rand()
    z_acc = np.random.rand()
    return x_acc, y_acc, z_acc

# Create 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize plot variables
x_data = []
y_data = []
z_data = []

# Create a 3D arrow or cone (choose your preferred shape)
#arrow = ax.quiver(0, 0, 0, 1, 0, 0, length=1, color='blue')  # Arrow
#arrow = ax.quiver(0, 0, 0, 1, 0, 0, length=1, color='blue')  # Arrow
#arrow = ax.plot_surface(X, Y, Z, color='blue', alpha=0.5)  # Cone (you'll need to define X, Y, Z)
# Create a 3D arrow
arrow = ax.plot([0, 1], [0, 0], [0, 0], color='blue')[0]  # Use Line3D object



# Real-time plotting loop
while True:
    x_acc, y_acc, z_acc = get_accelerometer_data()
    x_data.append(x_acc)
    y_data.append(y_acc)
    z_data.append(z_acc)

    # Update the arrow or cone based on the accelerometer data
    arrow.set_data(x_data[-1], y_data[-1])
    arrow.set_3d_properties(z_data[-1])
    # Update cone properties if using a cone

    plt.draw()
    plt.pause(0.01)  # Adjust pause time for real-time update rate

"""

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

# Define cone parameters
radius = 1
height = 2
angle = np.linspace(0, 2 * np.pi, 100)

# Create 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generate cone surface points
x = radius * np.outer(np.cos(angle), np.linspace(0, 1, 10))
y = radius * np.outer(np.sin(angle), np.linspace(0, 1, 10))
z = height * np.outer(np.ones(angle.shape), np.linspace(0, 1, 10))

# Create cone surface
cone = ax.plot_surface(x, y, z, color='blue', alpha=0.5)

# Real-time rotation loop
rotation_angle = 0

while True:
    # Rotate the cone
    x_rotated = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
    y_rotated = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)

    # Update the cone surface
    cone.set_data(x_rotated, y_rotated)
    cone.set_3d_properties(z)

    # Update the plot
    plt.draw()
    plt.pause(0.01)  # Adjust pause time for real-time update rate

    # Increment rotation angle
    rotation_angle += 0.01


"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = np.sin(-x*y)

fig = plt.figure()
ax = fig.gca()

ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

steps = 100
theta = np.linspace(0, 2 * np.pi, steps)
r_max = 1.2
x = np.zeros_like(theta)
y = r_max * np.cos(theta)
z = r_max * np.sin(theta)
ax.plot(x, y, z, 'r')
ax.plot(y, x, z, 'g')
ax.plot(z, y, x, 'b')

scale = 1.08
ax.quiver((0,), (0), (0), 
          (0), (0), (r_max), color=('c'))
ax.text(0, 0, r_max * scale, 'Z Theta', weight='bold')

ax.quiver((0), (0), (0), 
          (0), (r_max), (0), color=('m'))
ax.text(0, r_max * scale, 0, 'Y', weight='bold')

ax.quiver((0), (0), (0), 
          (r_max), (0), (0), color=('y'))
ax.text(r_max * scale, 0, 0, 'X', weight='bold')


plt.show()