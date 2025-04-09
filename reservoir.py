import numpy as np
import matplotlib.pyplot as plt

# Parameters
# Define constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
length = 2e-6  # Changed from 'len' to 'length'
width = 1.33e-7
height = 7.3e-8
A = length * width  # Area of the plates (m^2)
Ac = length * height
g0 = 1.5e-7  # Initial gap (m)
d0 = 5e-8  # Distance for coupling (m)
VDC = 1  # DC voltage (V)
VAC1 = 0.0001  # AC voltage 1 (V)
VAC2 = 0.0005  # AC voltage 2 (V)
m = 0.735*4.61356e-17  # Effective mass
mass = 4.61356e-17
f0 = 117.0818e6
omega0 = 2 * np.pi * f0  # Natural frequency in rad/s
# f1 = 117.0818e6
# omega = 2 * np.pi * f1
Q = 330  # Quality factor
alpha = 0.5
beta = 1.54e-4  # Constant for the cubic nonlinearity

# Time settings
t_min = 0  # Start time (s)
t_max = 5e-4  # End time (s)
dt = 1e-9  # Time step (s)
time = np.arange(t_min, t_max + dt, dt)
samples = len(time)

# Initial conditions
z1_0 = 0  # Initial displacement of resonator 1
v1_0 = 0  # Initial velocity of resonator 1
z2_0 = 0  # Initial displacement of resonator 2
v2_0 = 0  # Initial velocity of resonator 2

# Preallocate results
z1 = np.zeros(samples)
z2 = np.zeros(samples)
v1 = np.zeros(samples)
v2 = np.zeros(samples)

# Initial values
z1[0] = z1_0
z2[0] = z2_0
v1[0] = v1_0
v2[0] = v2_0
kc = -((epsilon_0 * A) / d0 ** 3) * (VAC2 - VAC1) ** 2

# Step input
steps = 1000
tau = (t_max - t_min) / steps
N = 400  # Size of mask
theta = tau / N  # Duration of each mask
mask = 0.45 + (np.random.rand(N)) * (0.75 - 0.45)  # Random mask values
mask = np.tile(mask, steps)
mask_time = np.linspace(t_min, t_max, steps * N)
feedback = np.zeros(N)
step_val = 2 * np.random.randint(0, 2, steps) - 1
step_time = np.linspace(t_min, t_max, steps)


def mask_function(t):
    return np.interp(t, mask_time, mask, left=mask[0], right=mask[-1])


def step_function(t):
    return np.interp(t, step_time, step_val, left=step_val[0], right=step_val[-1])


def feedback(t):
    if t < t_min + tau:
        return 0
    else:
        return alpha * np.interp(t - tau, time, v1, left=v1[0], right=v1[-1])


# Electric force functions
def F_elec1(t, y):
    return (epsilon_0 * A * (step_function(t) + feedback(t) + mask_function(t)  + VAC1) ** 2 * np.sin(omega0 * t) /
            (2 * (g0 - y[0]) ** 2))


def F_elec2(t, y):
    return 0


# Differential equations
def dydt(t, y):
    return np.array([
        y[1],
        F_elec1(t, y) / m - (omega0 * y[1] / Q) - (omega0 ** 2) * y[0] - (kc / m) * (y[0] - y[2]) - omega0**2 * beta * y[0]**3,
        y[3],
        F_elec2(t, y) / m - (omega0 * y[3] / Q) - (omega0 ** 2) * y[2] - (kc / m) * (y[2] - y[0]) - omega0**2 * beta * y[2]**3
    ])


def reservoir():
    # Runge-Kutta integration
    for i in range(samples - 1):
        t = time[i]
        y = np.array([z1[i], v1[i], z2[i], v2[i]])

        # Runge-Kutta steps
        k1 = dydt(t, y)
        k2 = dydt(t + dt / 2, y + dt * k1 / 2)
        k3 = dydt(t + dt / 2, y + dt * k2 / 2)
        k4 = dydt(t + dt, y + dt * k3)

        # Update solution
        y_next = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Store results
        z1[i + 1] = y_next[0]
        v1[i + 1] = y_next[1]
        z2[i + 1] = y_next[2]
        v2[i + 1] = y_next[3]

    # Plotting
    plt.figure(figsize=(10, 8))

    # Define number of samples to plot
    num_samples = min(5000, samples)

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(time[:num_samples], z1[:num_samples], 'b-', linewidth=2, label='v_1(t)')
    plt.plot(time[:num_samples], z2[:num_samples], 'r-', linewidth=1, label='v_2(t)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Displacement (m)', fontsize=12)
    plt.title('Time-Domain Response of Coupled NEMS Resonators', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)

    # Second subplot
    plt.subplot(2, 1, 2)
    num_step_samples = int((t_min + num_samples * dt) // tau) + 1
    plt.step(step_time[:num_step_samples], step_val[:num_step_samples],
             where='post', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Step Value')
    plt.title('Step Function')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return step_time, step_val, time, v1
