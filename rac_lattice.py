# -*- coding: utf-8 -*-
"""
Simplified 2D mesoscale model of Natural vs. Recycled Aggregate Concrete under
uniaxial compression -- pure Python version (no YADE needed).

Same idea as the YADE/CPM script: particles on a 2D lattice, three phases
(mortar / aggregate / ITZ), cohesive bonds between neighbours whose stiffness
and strength come from the phase factors, displacement-controlled compression,
and counting of broken bonds. Instead of explicit DEM dynamics it uses a
quasi-static bonded-particle (lattice) solver: at each load step the
equilibrium is solved with a sparse linear system, and overstressed bonds
(tension, shear with friction, or crushing) are broken one at a time, the most
critical first, until the network is stable again.

Requirements:  pip install numpy scipy matplotlib
Run:           python rac_lattice.py
"""

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. PARAMETERS
# ============================================================
young_mortar   = 25e9           # Pa
sigmaT_mortar  = 3.5e6          # Pa, tensile strength
frictionAngle  = math.atan(0.8)
shear_ratio    = 0.25           # shear/normal bond stiffness ratio
shear_strength = 2.0            # bond cohesion in shear, relative to tensile strength
compressive_ratio = 10.0        # bond compressive strength / tensile strength (concrete ~ 10)
strength_scatter = 0.2          # +/- 20% random bond strength (material heterogeneity)

factor_NA  = 2.0                # Natural Aggregate (stiff)
factor_RA  = 1.2                # Recycled Aggregate (softer, represents old mortar)
factor_ITZ = 0.6                # ITZ (weak interface)

width  = 0.10                   # m
height = 0.20                   # m
rMean  = 0.003                  # m
rFuzz  = 0.10                   # +/- radius variation
jitter = 0.05                   # position perturbation, fraction of rMean
thickness = 2 * rMean           # out-of-plane thickness for stress conversion
intRadius = 1.3                 # neighbour detection range for bonds

strainStep  = 1e-5              # axial strain increment per load step
finalStrain = 0.015             # hard stop
residualStop = 0.10             # also stop once stress falls below 10% of peak

MORTAR, NA, RA, ITZ = 0, 1, 2, 3
PHASE_NAMES = ['mortar', 'NA', 'RA', 'ITZ']


# ============================================================
# 2. GEOMETRY
# ============================================================
def make_specimen(recycled_fraction, seed=42):
    rng = np.random.default_rng(seed)
    dx, dy = 2 * rMean, math.sqrt(3) * rMean
    nx, ny = int(width / dx), int(height / dy)

    pos, rad, row_of = [], [], []
    for j in range(ny):
        y = -height / 2 + rMean + j * dy
        offset = rMean if j % 2 else 0.0
        for k in range(nx):
            x = -width / 2 + rMean + offset + k * dx
            if x > width / 2 - rMean + 1e-12:
                continue
            pos.append((x + jitter * rMean * (2 * rng.random() - 1),
                        y + jitter * rMean * (2 * rng.random() - 1)))
            rad.append(rMean * (1 + rFuzz * (2 * rng.random() - 1)))
            row_of.append(j)
    pos, rad, row_of = np.array(pos), np.array(rad), np.array(row_of)
    n = len(pos)

    # Aggregate zones (same geometry as the YADE script)
    x, y = pos[:, 0], pos[:, 1]
    in_agg = ((x**2 + (y - 0.06)**2) < 0.02**2) | \
             ((x**2 + (y + 0.06)**2) < 0.02**2) | \
             (((x - 0.04)**2 + y**2) < 0.015**2)
    phase = np.full(n, MORTAR)
    rand = rng.random(n)
    phase[in_agg] = np.where(rand[in_agg] < recycled_fraction, RA, NA)

    # Bonds between neighbours
    tree = cKDTree(pos)
    pairs = np.array(sorted(tree.query_pairs(r=intRadius * 2 * rad.max())))
    d = np.linalg.norm(pos[pairs[:, 1]] - pos[pairs[:, 0]], axis=1)
    keep = d < intRadius * (rad[pairs[:, 0]] + rad[pairs[:, 1]])
    pairs = pairs[keep]

    # ITZ: mortar particles bonded to an aggregate particle
    agg = np.isin(phase, [NA, RA])
    itz = np.zeros(n, bool)
    a, b = pairs[:, 0], pairs[:, 1]
    itz[a[agg[b] & (phase[a] == MORTAR)]] = True
    itz[b[agg[a] & (phase[b] == MORTAR)]] = True
    phase[itz] = ITZ

    bottom = np.where(row_of == 0)[0]
    top = np.where(row_of == ny - 1)[0]
    return pos, rad, phase, pairs, bottom, top, rng


def bond_properties(pos, rad, phase, pairs, rng):
    factor = np.array([1.0, factor_NA, factor_RA, factor_ITZ])
    E_p, sT_p = young_mortar * factor[phase], sigmaT_mortar * factor[phase]
    i, j = pairs[:, 0], pairs[:, 1]

    vec = pos[j] - pos[i]
    L = np.linalg.norm(vec, axis=1)
    nvec = vec / L[:, None]
    tvec = np.column_stack([-nvec[:, 1], nvec[:, 0]])

    E_b = 2 * E_p[i] * E_p[j] / (E_p[i] + E_p[j])       # series (harmonic) stiffness
    sT_b = np.minimum(sT_p[i], sT_p[j])                  # weakest phase governs
    sT_b *= 1 + strength_scatter * (2 * rng.random(len(pairs)) - 1)

    A = thickness * (rad[i] + rad[j])                    # bond cross-section
    kn = E_b * A / L
    ks = shear_ratio * kn
    Fn_max = sT_b * A
    Fs_max = shear_strength * Fn_max
    Fc_max = compressive_ratio * Fn_max
    return nvec, tvec, kn, ks, Fn_max, Fs_max, Fc_max


# ============================================================
# 3. SOLVER
# ============================================================
def assemble(n, pairs, nvec, tvec, kn, ks, alive):
    i, j = pairs[alive, 0], pairs[alive, 1]
    nv, tv, kn_, ks_ = nvec[alive], tvec[alive], kn[alive], ks[alive]
    # 2x2 bond stiffness: kn n n^T + ks t t^T
    k = (kn_[:, None, None] * nv[:, :, None] * nv[:, None, :] +
         ks_[:, None, None] * tv[:, :, None] * tv[:, None, :])
    dofs = np.column_stack([2 * i, 2 * i + 1, 2 * j, 2 * j + 1])
    sign = np.array([1, 1, -1, -1])
    ke = np.tile(k, (1, 2, 2)) * (sign[:, None] * sign[None, :])[None]
    rows = np.repeat(dofs, 4, axis=1).ravel()
    cols = np.tile(dofs, (1, 4)).ravel()
    K = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    # tiny regularisation so isolated fragments don't make K singular
    K = K + sp.identity(2 * n, format='csr') * (1e-9 * kn.mean())
    return K


def run_simulation(recycled_fraction, label):
    print(f"\n--- Running simulation: {label} (recycled fraction: {recycled_fraction}) ---")
    pos, rad, phase, pairs, bottom, top, rng = make_specimen(recycled_fraction)
    nvec, tvec, kn, ks, Fn_max, Fs_max, Fc_max = bond_properties(pos, rad, phase, pairs, rng)
    n, nb = len(pos), len(pairs)
    counts = ', '.join(f"{PHASE_NAMES[p]}: {np.sum(phase == p)}" for p in range(4))
    print(f"  particles: {n} ({counts}), bonds: {nb}")

    # Boundary conditions: bottom row fixed, top row moves down, both clamped in x
    fixed = np.concatenate([2 * bottom, 2 * bottom + 1, 2 * top, 2 * top + 1])
    free = np.setdiff1d(np.arange(2 * n), fixed)
    top_y = 2 * top + 1
    gauge = pos[top, 1].mean() - pos[bottom, 1].mean()
    tanphi = math.tan(frictionAngle)

    alive = np.ones(nb, bool)
    strain_vals, stress_vals, broken_vals = [], [], []
    peak, step = 0.0, 0

    while True:
        step += 1
        strain = step * strainStep
        u = np.zeros(2 * n)
        u[top_y] = -strain * gauge

        while True:   # break bonds one by one until equilibrium is stable
            K = assemble(n, pairs, nvec, tvec, kn, ks, alive)
            rhs = -K[free][:, fixed] @ u[fixed]
            u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)

            du = u.reshape(-1, 2)[pairs[:, 1]] - u.reshape(-1, 2)[pairs[:, 0]]
            Fn = kn * np.einsum('ij,ij->i', du, nvec)      # tension > 0
            Fs = ks * np.einsum('ij,ij->i', du, tvec)
            r_t = Fn / Fn_max
            r_s = np.abs(Fs) / (Fs_max + tanphi * np.maximum(-Fn, 0))
            r_c = -Fn / Fc_max                             # crushing
            r = np.where(alive, np.maximum(np.maximum(r_t, r_s), r_c), 0.0)
            worst = np.argmax(r)
            if r[worst] <= 1.0:
                break
            alive[worst] = False

        f = K @ u
        stress = -f[top_y].sum() / (width * thickness)
        strain_vals.append(strain)
        stress_vals.append(stress)
        broken_vals.append(int(nb - alive.sum()))
        peak = max(peak, stress)

        if step % 20 == 0:
            print(f"  strain {strain:.5f}  stress {stress/1e6:7.2f} MPa  broken {broken_vals[-1]}")
        if strain >= finalStrain or stress < residualStop * peak:
            break

    np.savetxt(f'{label}_stress_strain.txt',
               np.column_stack([strain_vals, stress_vals]), header='strain stress_Pa')
    np.savetxt(f'{label}_broken_bonds.txt',
               np.column_stack([strain_vals, broken_vals]), header='strain broken_bonds')
    print(f"  peak stress: {peak/1e6:.2f} MPa at strain {strain_vals[int(np.argmax(stress_vals))]:.5f}, "
          f"broken bonds at end: {broken_vals[-1]}")

    geometry = dict(pos=pos, phase=phase, pairs=pairs, alive=alive)
    return strain_vals, stress_vals, broken_vals, geometry


# ============================================================
# 4. RUN BOTH CASES
# ============================================================
strain_nat, stress_nat, broken_nat, geo_nat = run_simulation(0.0, 'natural')
strain_rec, stress_rec, broken_rec, geo_rec = run_simulation(0.5, 'recycled')

# ============================================================
# 5. PLOTS
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

# Crack patterns: specimen phases with broken bonds drawn in red
colors = np.array(['#bdbdbd', '#4a6fa5', '#c98a3a', '#7fbf7f'])
fig, axes = plt.subplots(1, 2, figsize=(8, 8))
for ax, geo, title in [(axes[0], geo_nat, 'Natural (0% RA)'), (axes[1], geo_rec, 'Recycled (50% RA)')]:
    p = geo['pos']
    ax.scatter(p[:, 0], p[:, 1], s=18, c=colors[geo['phase']], linewidths=0)
    for a, b in geo['pairs'][~geo['alive']]:
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], 'r-', linewidth=1.5)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nbroken bonds: {np.sum(~geo["alive"])}')
    ax.set_xticks([]); ax.set_yticks([])
handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=nm) for c, nm in zip(colors, PHASE_NAMES)]
handles.append(plt.Line2D([], [], color='r', label='broken bond'))
fig.legend(handles=handles, loc='lower center', ncol=5)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig('rac_crack_patterns.png', dpi=150)
plt.close()

print("\nSimulation complete. Files created:")
print("  natural_stress_strain.txt, recycled_stress_strain.txt")
print("  natural_broken_bonds.txt, recycled_broken_bonds.txt")
print("  rac_comparison.png, rac_crack_patterns.png")
