Re-Entry Simulator

A Python simulator that models a spacecraft re-entering Earth's atmosphere as a point mass, integrated with 4th-order Runge-Kutta (RK4). Built in stages, starting simple and adding physics over time.

Versions
v1 — reentry_sim.py

A basic ballistic entry model:

Gravity (falls off with altitude) + drag only, no lift
Simple exponential atmosphere (ρ = ρ₀ · e^(-h/H))
Outputs: altitude, velocity, deceleration (g's), and trajectory shape

Plot: reentry_plots.png

v2 — reentry_sim_v2.py

Builds on v1 with three additions:

Lift — vehicle now has an L/D ratio, so aerodynamic lift bends the flight path instead of gravity/drag alone
Aerodynamic heating — Sutton-Graves equation estimates stagnation-point heat flux, plus cumulative heat load over the entry
Layered atmosphere — 8-layer piecewise model (troposphere/stratosphere/mesosphere with real lapse rates) instead of a single exponential curve, more accurate below ~86 km

Plot: reentry_plots_v2.png

Requirements
pip install numpy matplotlib
Usage
python reentry_sim.py       # v1, ballistic only
python reentry_sim_v2.py    # v2, with lift + heating + layered atmosphere

Each script prints a summary (flight time, final velocity, peak deceleration, downrange, and for v2, peak heat flux and total heat load) and saves its plots as a PNG in the same folder.

Key parameters to tweak

Both scripts expose vehicle and entry parameters near the top of the file:

Parameter	Description
MASS	Vehicle mass (kg)
DRAG_COEFF	Drag coefficient (Cd)
AREA	Reference cross-sectional area (m²)
INITIAL_ALTITUDE	Entry interface altitude (m)
INITIAL_VELOCITY	Entry velocity (m/s)
INITIAL_FLIGHT_PATH_ANGLE_DEG	Entry angle below horizontal (degrees)

v2 adds:

Parameter	Description
LIFT_TO_DRAG	L/D ratio — 0 for pure ballistic, ~0.24 for Apollo-like, 1+ for lifting body
NOSE_RADIUS	Effective nose radius (m) — smaller = sharper vehicle = higher peak heating
Roadmap

Possible next steps for a v3:

Mach-dependent drag/lift coefficients (Cd/Cl curves instead of constants)
Full 6-DOF (vehicle orientation, angle of attack, guidance/control)
Parachute or retro-propulsion phase after main entry heating ends