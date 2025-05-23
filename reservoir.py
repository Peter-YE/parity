import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

# Parameters
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
m = 0.735 * 4.61356e-17  # Effective mass
mass = 4.61356e-17
f0 = 117.0818e6
omega0 = 2 * np.pi * f0  # Natural frequency in rad/s
Q = 330  # Quality factor
alpha = 0.5
beta = 1.54e-4  # Constant for the cubic nonlinearity


# Time settings
t_min = 0  # Start time (s)
t_max = 5e-4  # End time (s)
dt = 1e-9  # Time step (s) i.e. sampling rate
time = np.arange(t_min, t_max + dt, dt)
time = time[1:]  # Remove the first element to match the size of other arrays
n_samples = len(time)


# Initial conditions
z1_0 = 0  # Initial displacement of resonator 1
v1_0 = 0  # Initial velocity of resonator 1
z2_0 = 0  # Initial displacement of resonator 2
v2_0 = 0  # Initial velocity of resonator 2

# Preallocate results
z1 = np.zeros(n_samples)
z2 = np.zeros(n_samples)
v1 = np.zeros(n_samples)
v2 = np.zeros(n_samples)

# Initial values
z1[0] = z1_0
z2[0] = z2_0
v1[0] = v1_0
v2[0] = v2_0
kc = -((epsilon_0 * A) / (d0 ** 3)) * ((VAC2 - VAC1) ** 2)
# kc=1

"""Step input"""
n_steps = 1000
step_val = 2 * np.random.randint(0, 2, n_steps) - 1
# The following creates non-random steps input
# step_val = np.zeros(n_steps)
# for i in range(n_steps):
#     if i % 2 == 0:
#         step_val[i] = 1
#     else:
#         step_val[i] = -1
step_time = np.linspace(t_min, t_max, n_steps)
step_val_real = np.repeat(step_val, n_samples // n_steps)

"""mask function"""
n_mask = 100  # Size of mask per step
tau = (t_max - t_min) / n_steps
theta = tau / n_mask  # Duration of each mask
mask = 0.45 + (np.random.rand(n_mask)) * (0.75 - 0.45)  # Random mask values
mask = np.tile(mask, n_steps)
mask_time = np.linspace(t_min, t_max, n_steps * n_mask)
mask_real = np.repeat(mask, n_samples // (n_mask * n_steps))
print("time size", np.size(time))
print("step size", np.size(step_val_real))


"""real time values"""
def mask_function(t):
    if t < t_min + tau:
        return mask_real[0]
    else:
        index = int((t - t_min) // dt)
        return mask_real[index]


def step_function(t):
    index = int((t - t_min) // dt)
    return step_val_real[index]


def feedback(t):
    if t < t_min + tau:
        return v1[0]
    else:
        index = int((t - tau - t_min) // dt)
        return v1[index]

"""Forcing functions"""
def F_elec1(t, y):
    return (epsilon_0 * A * ((step_function(t) + mask_function(t) + feedback(t)) + VAC1 * np.sin(omega0 * t)) ** 2 /
            (2 * (g0 - y[0]) ** 2))
    # return (epsilon_0 * A * (VDC+(step_function(t) + feedback(t) + mask_function(t))*np.sin(omega0 * t)) ** 2 /
    #         (2 * (g0 - y[0]) ** 2))


def F_elec2(t, y):
    if t > tau:
        return (epsilon_0 * A * ((step_function(t - tau) + mask_function(t - tau)) + VAC2 * np.sin(omega0 * t)) ** 2 /
                (2 * (g0 - y[0]) ** 2))
    else:
        return (epsilon_0 * A * ((step_function(t) + mask_function(t)) + VAC2 * np.sin(omega0 * t)) ** 2 /
                (2 * (g0 - y[0]) ** 2))


"""Differential equations"""
def dydt(t, y):
    return np.array([
        y[1],
        F_elec1(t, y) / m - (omega0 * y[1] / Q) - (omega0 ** 2) * y[0] - (kc / m) * (y[0] - y[2]) - omega0 ** 2 * beta * y[0] ** 3,
        y[3],
        F_elec2(t, y) / m - (omega0 * y[3] / Q) - (omega0 ** 2) * y[2] - (kc / m) * (y[2] - y[0]) - omega0 ** 2 * beta * y[2] ** 3
    ])

"""Reservoir simulation"""
def reservoir():
    # Runge-Kutta integration
    for i in range(n_samples - 1):
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
    plot_min = 7000
    plot_max = 10000

    mpl.rcParams['font.family'] = 'Times New Roman'

    # First subplot
    plt.subplot(2, 1, 2)
    plt.plot(time[plot_min:plot_max], v1[plot_min:plot_max], linewidth=2, label='v₁(t)')
    plt.xlabel('Time (s)', fontsize=28)
    plt.ylabel('Displacement\nVelocity (m/s)', fontsize=28)  # Changed from Velocity (m/s)
    plt.title('Time-Domain Response of NEMS Resonators', fontsize=29)
    plt.tick_params(axis='both', labelsize=28)

    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis='x', style='sci', scilimits=(-5, -5))
    ax1.xaxis.offsetText.set_fontsize(28)
    ax1.yaxis.offsetText.set_fontsize(28)
    # plt.grid(True)

    # Second subplot
    plt.subplot(2, 1, 1)
    # num_min = int(num_step_samples_min*N)
    # num_max = int(num_step_samples_max*N)
    # plot mask
    # mask_time = np.linspace(t_min, t_max, num_step_samples_max * N)
    # plt.step(mask_time[num_min:num_max],mask[num_min:num_max])
    plt.step(time[plot_min:plot_max], step_val_real[plot_min:plot_max], where='post', linewidth=1.5)

    plt.xlabel('Time (s)', fontsize=28)
    plt.ylabel('Step Value', fontsize=28)
    plt.title('Step Input Function', fontsize=29)
    plt.tick_params(axis='both', labelsize=28)

    ax2 = plt.gca()
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-5, -5))
    ax2.xaxis.offsetText.set_fontsize(28)
    ax2.yaxis.offsetText.set_fontsize(28)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    return step_time, step_val, time, v1
