# -*- coding: utf-8 -*-
"""
Simplified 2D mesoscale DEM model of Natural vs. Recycled Aggregate Concrete
under uniaxial compression. Based on Mei et al. (2024) framework, simplified
to a 2D three-phase representation (mortar / aggregate / ITZ).

Run from a terminal (no GUI, exit when done):
    yade -n -x rac_demo.py
"""

from yade import plot, pack, utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

# ============================================================
# 1. PARAMETERS
# ============================================================
# New Mortar (NM)
young_mortar   = 25e9           # Pa
poisson_mortar = 0.2
sigmaT_mortar  = 3.5e6          # Pa, tensile strength
frictionAngle  = math.atan(0.8)
epsCrackOnset  = 1e-4           # CPM: strain at crack onset (required, default is NaN)
relDuctility   = 30.            # CPM: relative ductility (required, default is NaN)

# Phase factors relative to mortar
factor_NA  = 2.0                # Natural Aggregate (stiff)
factor_RA  = 1.2                # Recycled Aggregate (softer, represents old mortar)
factor_ITZ = 0.6                # ITZ (weak interface)

# Specimen (2D, one particle layer in the x-y plane)
width  = 0.10                   # m
height = 0.20                   # m
rMean  = 0.003                  # m
rFuzz  = 0.10                   # +/- 10% radius variation (kept small so the lattice bonds cleanly)
jitter = 0.05                   # random position perturbation, fraction of rMean
thickness = 2 * rMean           # out-of-plane thickness used to convert force -> stress

intRadius = 1.3                 # bond detection range at start (creates cohesive CPM bonds)

# Loading
strainRate  = 0.5               # 1/s  (high for concrete; lower it for more quasi-static results)
finalStrain = 0.015             # stop at 1.5% axial strain
recordEvery = 100               # record data every N iterations

# ============================================================
# 2. GLOBAL STATE USED BY THE PyRunner CALLBACK
#    (PyRunner executes in the module namespace, so recordData
#     must be a top-level function, not nested inside another one)
# ============================================================
top_ids, bottom_ids = [], []
top_y0 = 0.0
gauge_length = 1.0
initial_bonds = None
strain_vals, stress_vals, broken_vals = [], [], []


def count_cohesive_bonds():
    n = 0
    for i in O.interactions:
        if i.isReal and getattr(i.phys, 'isCohesive', False):
            n += 1
    return n


def recordData():
    if initial_bonds is None:
        return
    topY = O.bodies[top_ids[0]].state.pos[1]
    strain = (top_y0 - topY) / gauge_length
    # Reaction force of the specimen on the top clamped particles (compression > 0)
    force = sum(O.forces.f(i)[1] for i in top_ids)
    stress = force / (width * thickness)
    broken = initial_bonds - count_cohesive_bonds()

    strain_vals.append(strain)
    stress_vals.append(stress)
    broken_vals.append(broken)

    if strain >= finalStrain:
        O.pause()


def make_hex_packing(material, rng):
    """2D hexagonal particle arrangement with small random size/position variation."""
    dx = 2 * rMean
    dy = math.sqrt(3) * rMean
    nx = int(width / dx)
    ny = int(height / dy)
    rows = []
    for j in range(ny):
        y = -height / 2 + rMean + j * dy
        offset = rMean if j % 2 else 0.0
        row = []
        for k in range(nx):
            x = -width / 2 + rMean + offset + k * dx
            if x > width / 2 - rMean + 1e-12:
                continue
            r = rMean * (1 + rFuzz * (2 * rng.random() - 1))
            px = x + jitter * rMean * (2 * rng.random() - 1)
            py = y + jitter * rMean * (2 * rng.random() - 1)
            row.append(O.bodies.append(utils.sphere((px, py, 0), r, material=material)))
        rows.append(row)
    return rows


# ============================================================
# 3. ONE SIMULATION FOR A GIVEN RECYCLED FRACTION
# ============================================================
def run_simulation(recycled_fraction, label):
    global top_ids, bottom_ids, top_y0, gauge_length, initial_bonds
    global strain_vals, stress_vals, broken_vals

    print(f"\n--- Running simulation: {label} (recycled fraction: {recycled_fraction}) ---")
    O.reset()
    rng = np.random.default_rng(42)   # same geometry/randomness for both cases

    def cpm(factor, density):
        return CpmMat(young=young_mortar * factor, poisson=poisson_mortar,
                      sigmaT=sigmaT_mortar * factor, frictionAngle=frictionAngle,
                      epsCrackOnset=epsCrackOnset, relDuctility=relDuctility,
                      density=density)

    # Keep the Material objects (not just their ids) so we can assign them to bodies
    matMortar = cpm(1.0, 2400); O.materials.append(matMortar)
    matNA     = cpm(factor_NA, 2700); O.materials.append(matNA)
    matRA     = cpm(factor_RA, 2400); O.materials.append(matRA)
    matITZ    = cpm(factor_ITZ, 2200); O.materials.append(matITZ)

    rows = make_hex_packing(matMortar, rng)

    # --- Aggregates: three circular zones ---
    aggregates = []
    for b in O.bodies:
        x, y = b.state.pos[0], b.state.pos[1]
        if (x**2 + (y - 0.06)**2) < 0.02**2 or \
           (x**2 + (y + 0.06)**2) < 0.02**2 or \
           ((x - 0.04)**2 + y**2) < 0.015**2:
            b.material = matRA if rng.random() < recycled_fraction else matNA
            aggregates.append(b)

    # --- ITZ: mortar particles that are direct neighbours of an aggregate particle ---
    n_itz = 0
    for b in O.bodies:
        if b.material is not matMortar:
            continue
        for a in aggregates:
            if (b.state.pos - a.state.pos).norm() < intRadius * (b.shape.radius + a.shape.radius):
                b.material = matITZ
                n_itz += 1
                break
    print(f"  particles: {len(O.bodies)}, aggregate: {len(aggregates)}, ITZ: {n_itz}")

    # --- 2D constraint: no out-of-plane motion or tilting ---
    for b in O.bodies:
        b.state.blockedDOFs = 'zXY'

    # --- Loading via clamped end rows (instead of walls) ---
    bottom_ids = list(rows[0])
    top_ids = list(rows[-1])
    for i in bottom_ids + top_ids:
        O.bodies[i].state.blockedDOFs = 'xyzXYZ'
    top_y0 = O.bodies[top_ids[0]].state.pos[1]
    gauge_length = top_y0 - O.bodies[bottom_ids[0]].state.pos[1]

    # --- Engines ---
    O.engines = [
        ForceResetter(),
        InsertionSortCollider([Bo1_Sphere_Aabb(aabbEnlargeFactor=intRadius, label='is2aabb')]),
        InteractionLoop(
            [Ig2_Sphere_Sphere_ScGeom(interactionDetectionFactor=intRadius, label='ss2sc')],
            [Ip2_CpmMat_CpmMat_CpmPhys()],
            [Law2_ScGeom_CpmPhys_Cpm()]
        ),
        NewtonIntegrator(damping=0.4),
        PyRunner(command='recordData()', iterPeriod=recordEvery),
    ]

    # Stable time step (a fixed 1e-6 s is too large for these stiffnesses)
    O.dt = 0.5 * utils.PWaveTimeStep()

    # First step creates the cohesive bonds, then restore normal contact detection
    O.step()
    is2aabb.aabbEnlargeFactor = 1.0
    ss2sc.interactionDetectionFactor = 1.0

    strain_vals, stress_vals, broken_vals = [], [], []
    initial_bonds = count_cohesive_bonds()
    print(f"  initial cohesive bonds: {initial_bonds}, dt = {O.dt:.3e} s")

    # Top row moves down, bottom row stays fixed
    v = strainRate * gauge_length
    for i in top_ids:
        O.bodies[i].state.vel = Vector3(0, -v, 0)

    # O.run() returns immediately; O.wait() blocks until recordData() pauses it
    O.run()
    O.wait()

    np.savetxt(f'{label}_stress_strain.txt',
               np.column_stack([strain_vals, stress_vals]), header='strain stress_Pa')
    np.savetxt(f'{label}_broken_bonds.txt',
               np.column_stack([strain_vals, broken_vals]), header='strain broken_bonds')
    print(f"  peak stress: {max(stress_vals)/1e6:.2f} MPa, broken bonds at end: {broken_vals[-1]}")

    return list(strain_vals), list(stress_vals), list(broken_vals)


# ============================================================
# 4. RUN BOTH CASES
# ============================================================
strain_nat, stress_nat, broken_nat = run_simulation(0.0, 'natural')
strain_rec, stress_rec, broken_rec = run_simulation(0.5, 'recycled')

# ============================================================
# 5. PLOT RESULTS
# ============================================================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(strain_nat, np.array(stress_nat) / 1e6, 'b-', linewidth=2, label='Natural (0% RA)')
plt.plot(strain_rec, np.array(stress_rec) / 1e6, 'r--', linewidth=2, label='Recycled (50% RA)')
plt.xlabel('Axial Strain')
plt.ylabel('Stress (MPa)')
plt.title('Stress-Strain Curves')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(strain_nat, broken_nat, 'b-', linewidth=2, label='Natural (0% RA)')
plt.plot(strain_rec, broken_rec, 'r--', linewidth=2, label='Recycled (50% RA)')
plt.xlabel('Axial Strain')
plt.ylabel('Number of Broken Bonds (Cracks)')
plt.title('Crack Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('rac_comparison.png', dpi=150)
plt.close()

print("\nSimulation complete. Files created:")
print("  natural_stress_strain.txt, recycled_stress_strain.txt")
print("  natural_broken_bonds.txt, recycled_broken_bonds.txt")
print("  rac_comparison.png")
