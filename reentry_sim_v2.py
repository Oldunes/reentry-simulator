"""
Re-Entry Simulator v2 — Lift, Heating, and a Layered Atmosphere + v1 (drag-only) model

  1. LIFT — vehicle has an L/D ratio, so aerodynamic lift alos impacts flight path angle instead of just gravity + drag
  2. AERODYNAMIC HEATING — Sutton-Graves equation estimates stagnation-point heat flux and cumulative heat load (total energy absorbed).
  3. LAYERED ATMOSPHERE — a simplified piecewise US Standard Atmosphere (troposphere/stratosphere/mesosphere) instead of one flat exponential curve, for more realistic density at low altitude.

State vector: [altitude, downrange, velocity, flight_path_angle]
  altitude - height above surface (m)
  downrange - horizontal distance traveled (m)
  velocity - speed along flight path (m/s)
  flight_path_angle - angle below local horizontal (rad), negative = descending

Integrated with 4th-order Runge-Kutta (RK4).
"""
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
G0 = 9.80665            # sea-level gravitational acceleration (m/s^2)
EARTH_RADIUS = 6.371e6  # mean Earth radius (m)
GAS_CONSTANT_AIR = 287.05  # specific gas constant for dry air (J/(kg*K))

# VEHICLE PARAMETERS — tweak these to try different entry vehicles
MASS = 500.0             # kg
DRAG_COEFF = 1.2         # dimensionless drag coefficient (Cd)
AREA = 1.0                # reference cross-sectional area (m^2)
LIFT_TO_DRAG = 0.3        # L/D ratio (0 = pure ballistic, ~0.24 for Apollo, ~1+ for Shuttle)
NOSE_RADIUS = 0.3         # effective nose radius for heating calc (m)


# INITIAL CONDITIONS
INITIAL_ALTITUDE = 120_000.0          # m (typical re-entry interface altitude)
INITIAL_VELOCITY = 7800.0             # m/s (roughly LEO orbital velocity)
INITIAL_FLIGHT_PATH_ANGLE_DEG = -6.0  # degrees below horizontal (entry angle)

DT = 0.05          # integration timestep (s)
MAX_TIME = 2000.0  # safety cutoff (s)

# Sutton-Graves heating coefficient (empirical, for air, Rn in meters)
SUTTON_GRAVES_K = 1.7415e-4

# ATMOSPHERE — simplified layered model (piecewise, based on the ideal gas
# law + standard lapse rates for troposphere / stratosphere / mesosphere).
# More accurate than a single exponential, especially below ~50 km.

# Each layer: (base_altitude_m, base_temp_K, lapse_rate_K_per_m, base_pressure_Pa)
ATMOSPHERE_LAYERS = [
    (0.0,     288.15, -0.0065, 101325.0),
    (11000.0, 216.65,  0.0,     22632.1),
    (20000.0, 216.65,  0.001,    5474.9),
    (32000.0, 228.65,  0.0028,   868.02),
    (47000.0, 270.65,  0.0,      110.91),
    (51000.0, 270.65, -0.0028,    66.94),
    (71000.0, 214.65, -0.002,     3.96),
    (86000.0, 186.87,  0.0,       0.3734),
]


def atmospheric_density(altitude):
    """
    "Layered atmosphere model (approximates US Standard Atmosphere 1976 up to ~86 km). Above that, falls back to a thin exponential tail so the sim doesn't blow up at high altitude."
    """
    if altitude < 0:
        altitude = 0.0

    if altitude > 86000.0:
        # Thin exponential tail for the upper mesosphere/thermosphere
        rho_86 = 6.958e-6  # approx density at 86 km (kg/m^3)
        return rho_86 * np.exp(-(altitude - 86000.0) / 6000.0)

    # Find which layer this altitude falls in
    layer = ATMOSPHERE_LAYERS[0]
    for i in range(len(ATMOSPHERE_LAYERS)):
        if altitude >= ATMOSPHERE_LAYERS[i][0]:
            layer = ATMOSPHERE_LAYERS[i]
        else:
            break

    base_alt, base_temp, lapse_rate, base_pressure = layer
    d_alt = altitude - base_alt

    if abs(lapse_rate) < 1e-9:
        # Isothermal layer
        temp = base_temp
        pressure = base_pressure * np.exp(-G0 * d_alt / (GAS_CONSTANT_AIR * base_temp))
    else:
        temp = base_temp + lapse_rate * d_alt
        pressure = base_pressure * (temp / base_temp) ** (-G0 / (GAS_CONSTANT_AIR * lapse_rate))

    density = pressure / (GAS_CONSTANT_AIR * temp)
    return max(density, 0.0)


def gravity(altitude):
    """Gravity falls off with altitude (inverse square)."""
    r = EARTH_RADIUS + altitude
    return G0 * (EARTH_RADIUS / r) ** 2


def heat_flux(altitude, velocity):
    """
    Sutton-Graves stagnation-point convective heat flux (W/m^2).
    q = k * sqrt(rho / Rn) * v^3
    """
    rho = atmospheric_density(altitude)
    if rho <= 0 or velocity <= 0:
        return 0.0
    return SUTTON_GRAVES_K * np.sqrt(rho / NOSE_RADIUS) * velocity ** 3


def derivatives(state):
    """
    Compute d(state)/dt for the point-mass re-entry model, now including lift.
    state = [altitude, downrange, velocity, flight_path_angle]
    """
    altitude, downrange, velocity, gamma = state

    rho = atmospheric_density(altitude)
    g = gravity(altitude)

    drag = 0.5 * rho * velocity ** 2 * DRAG_COEFF * AREA
    lift = drag * LIFT_TO_DRAG  # lift derived from drag via L/D ratio

    drag_decel = drag / MASS
    lift_accel = lift / MASS

    # Equations of motion (with lift):
    d_altitude = velocity * np.sin(gamma)
    d_downrange = velocity * np.cos(gamma)
    d_velocity = -drag_decel - g * np.sin(gamma)
    d_gamma = (lift_accel - g * np.cos(gamma)) / velocity if velocity > 1e-6 else 0.0

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

    times, altitudes, downranges, velocities = [], [], [], []
    decel_g, heat_fluxes, cumulative_heat = [], [], []

    t = 0.0
    total_heat = 0.0  # J/m^2, integrated heat load
    while state[0] > 0 and t < MAX_TIME:
        altitude, downrange, velocity, gamma = state

        rho = atmospheric_density(altitude)
        drag = 0.5 * rho * velocity ** 2 * DRAG_COEFF * AREA
        decel_in_gs = (drag / MASS) / G0
        q = heat_flux(altitude, velocity)

        total_heat += q * DT  # simple rectangular integration

        times.append(t)
        altitudes.append(altitude)
        downranges.append(downrange)
        velocities.append(velocity)
        decel_g.append(decel_in_gs)
        heat_fluxes.append(q / 1e6)          # convert to MW/m^2 for readability
        cumulative_heat.append(total_heat / 1e6)  # MJ/m^2

        state = rk4_step(state, DT)
        t += DT

    return {
        "t": np.array(times),
        "altitude": np.array(altitudes),
        "downrange": np.array(downranges),
        "velocity": np.array(velocities),
        "decel_g": np.array(decel_g),
        "heat_flux": np.array(heat_fluxes),
        "cumulative_heat": np.array(cumulative_heat),
    }


def plot_results(results):
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    fig.suptitle("Re-Entry Simulation v2 — Lift + Heating + Layered Atmosphere",
                 fontsize=14, fontweight="bold")

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
    ax.set_title(f"Trajectory (L/D = {LIFT_TO_DRAG})")
    ax.grid(True, alpha=0.3)

    # Heat flux vs Time
    ax = axes[2, 0]
    ax.plot(results["t"], results["heat_flux"], color="tab:purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat Flux (MW/m^2)")
    ax.set_title("Stagnation-Point Heat Flux vs Time")
    ax.grid(True, alpha=0.3)
    max_q = results["heat_flux"].max()
    max_q_t = results["t"][results["heat_flux"].argmax()]
    ax.annotate(f"Peak: {max_q:.1f} MW/m^2", xy=(max_q_t, max_q),
                xytext=(max_q_t + 5, max_q * 0.9),
                arrowprops=dict(arrowstyle="->", color="black"))

    # Cumulative heat load vs Time
    ax = axes[2, 1]
    ax.plot(results["t"], results["cumulative_heat"], color="tab:brown")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Heat Load (MJ/m^2)")
    ax.set_title("Total Heat Load vs Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reentry_plots_v2.png", dpi=150)
    print("Saved plots to reentry_plots_v2.png")


if __name__ == "__main__":
    results = run_simulation()

    print(f"Simulation complete.")
    print(f"  Flight time: {results['t'][-1]:.1f} s")
    print(f"  Final velocity: {results['velocity'][-1]:.1f} m/s")
    print(f"  Peak deceleration: {results['decel_g'].max():.1f} g")
    print(f"  Peak heat flux: {results['heat_flux'].max():.1f} MW/m^2")
    print(f"  Total heat load: {results['cumulative_heat'][-1]:.1f} MJ/m^2")
    print(f"  Total downrange: {results['downrange'][-1] / 1000:.1f} km")

    plot_results(results)
