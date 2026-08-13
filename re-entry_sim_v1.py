"""
Simple Ballistic Re-Entry Simulator
Models a re-entry vehicle as a point mass falling through an exponential atmosphere, subject to gravity and drag (no lift yet — that's a good next step once you're ready to add complexity).

State vector: [altitude, downrange, velocity, flight_path_angle]
  altitude - height above surface (m)
  downrange - horizontal distance traveled (m)
  velocity - speed along flight path (m/s)
  flight_path_angle - angle below local horizontal (rad), negative = descending

Integrated with 4th-order Runge-Kutta (RK4) for decent accuracy without
needing a stiff-ODE solver.
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
G0 = 9.80665          # sea-level gravitational acceleration (m/s^2)
EARTH_RADIUS = 6.371e6  # mean Earth radius (m)
RHO0 = 1.225           # sea-level air density (kg/m^3)
SCALE_HEIGHT = 8500.0  # atmospheric scale height (m), exponential model


# vehicle parameters (can be changed to simulate different vehicles)
MASS = 500.0            # kg
DRAG_COEFF = 1.2        # dimensionless drag coefficient (Cd)
AREA = 1.0               # cross-sectional area (m^2)

# initial conditions
INITIAL_ALTITUDE = 120_000.0        # meters (m) (typical re-entry interface altitude)
INITIAL_VELOCITY = 7800.0           # m/s (roughly LEO orbital velocity)
INITIAL_FLIGHT_PATH_ANGLE_DEG = -6.0  # degrees below horizontal (entry angle)

DT = 0.05     # integration timestep (s)
MAX_TIME = 2000.0  # safety cutoff (s)


def atmospheric_density(altitude):
    """Exponential atmosphere model. Returns density in kg/m^3."""
    if altitude < 0:
        altitude = 0
    return RHO0 * np.exp(-altitude / SCALE_HEIGHT)


def gravity(altitude):
    """Gravity falls off with altitude (inverse square)."""
    r = EARTH_RADIUS + altitude
    return G0 * (EARTH_RADIUS / r) ** 2


def derivatives(state):
    """
    Compute d(state)/dt for the point-mass re-entry model.
    state = [altitude, downrange, velocity, flight_path_angle]
    """
    altitude, downrange, velocity, gamma = state

    rho = atmospheric_density(altitude)
    g = gravity(altitude)

    # Drag force (opposes velocity direction)
    drag = 0.5 * rho * velocity ** 2 * DRAG_COEFF * AREA
    drag_decel = drag / MASS

    # Equations of motion (no lift):
    d_altitude = velocity * np.sin(gamma)
    d_downrange = velocity * np.cos(gamma)
    d_velocity = -drag_decel - g * np.sin(gamma)
    d_gamma = -g * np.cos(gamma) / velocity if velocity > 1e-6 else 0.0

    return np.array([d_altitude, d_downrange, d_velocity, d_gamma])


def rk4_step(state, dt):
    """Single RK4 integration step."""
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_simulation():
    gamma0 = np.radians(INITIAL_FLIGHT_PATH_ANGLE_DEG)
    state = np.array([INITIAL_ALTITUDE, 0.0, INITIAL_VELOCITY, gamma0])

    times, altitudes, downranges, velocities, decel_g = [], [], [], [], []

    t = 0.0
    while state[0] > 0 and t < MAX_TIME:
        rho = atmospheric_density(state[0])
        drag = 0.5 * rho * state[2] ** 2 * DRAG_COEFF * AREA
        decel_in_gs = (drag / MASS) / G0

        times.append(t)
        altitudes.append(state[0])
        downranges.append(state[1])
        velocities.append(state[2])
        decel_g.append(decel_in_gs)

        state = rk4_step(state, DT)
        t += DT

    return {
        "t": np.array(times),
        "altitude": np.array(altitudes),
        "downrange": np.array(downranges),
        "velocity": np.array(velocities),
        "decel_g": np.array(decel_g),
    }


def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Ballistic Re-Entry Simulation", fontsize=14, fontweight="bold")

    # Altitude vs Time
    ax = axes[0, 0]
    ax.plot(results["t"], results["altitude"] / 1000, color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs Time")
    ax.grid(True, alpha=0.3)

    # Velocity vs Time
    ax = axes[0, 1]
    ax.plot(results["t"], results["velocity"], color="tab:red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity vs Time")
    ax.grid(True, alpha=0.3)

    # Deceleration (g's) vs Time
    ax = axes[1, 0]
    ax.plot(results["t"], results["decel_g"], color="tab:orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Deceleration (g)")
    ax.set_title("Deceleration vs Time")
    ax.grid(True, alpha=0.3)
    max_g = results["decel_g"].max()
    max_g_t = results["t"][results["decel_g"].argmax()]
    ax.annotate(f"Peak: {max_g:.1f} g", xy=(max_g_t, max_g),
                xytext=(max_g_t + 5, max_g * 0.9),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Altitude vs Downrange (trajectory shape)
    ax = axes[1, 1]
    ax.plot(results["downrange"] / 1000, results["altitude"] / 1000, color="tab:green")
    ax.set_xlabel("Downrange (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Trajectory (Altitude vs Downrange)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reentry_plots.png", dpi=150)
    print("Saved plots to reentry_plots.png")


if __name__ == "__main__":
    results = run_simulation()

    print(f"Simulation complete.")
    print(f"  Flight time: {results['t'][-1]:.1f} s")
    print(f"  Final velocity: {results['velocity'][-1]:.1f} m/s")
    print(f"  Peak deceleration: {results['decel_g'].max():.1f} g")
    print(f"  Total downrange: {results['downrange'][-1] / 1000:.1f} km")

    plot_results(results)
