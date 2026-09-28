import openmm
import openmm.app

# Dimensionless Units

length_scale = openmm.unit.nanometer
mass_scale = openmm.unit.dalton
energy_scale = openmm.unit.kilojoule_per_mole
time_scale = openmm.unit.picosecond

press_scale = energy_scale / (length_scale**3 * openmm.unit.AVOGADRO_CONSTANT_NA)
temp_scale = energy_scale / (
    openmm.unit.BOLTZMANN_CONSTANT_kB * openmm.unit.AVOGADRO_CONSTANT_NA
)

# Simulation Parameters

N_A = 500
N_B = 500
rho = 0.6 / length_scale**3
m_A = 1.0 * mass_scale
m_B = 1.1 * mass_scale

r_cut = 3.0 * length_scale
sigma_A = 1.0 * length_scale
sigma_B = 1.2 * length_scale
epsilon_A = 1.0 * energy_scale
epsilon_B = 1.3 * energy_scale

press = 2.0 * press_scale
temp = 1.5 * temp_scale
friction = 1.0 / time_scale
step = 0.005 * time_scale

# System Creation

system = openmm.System()

L_box = ((N_A + N_B) / rho) ** (1 / 3)
system.setDefaultPeriodicBoxVectors(
    openmm.Vec3(L_box, 0, 0), openmm.Vec3(0, L_box, 0), openmm.Vec3(0, 0, L_box)
)

for i_A in range(N_A):
    system.addParticle(m_A)
for i_B in range(N_B):
    system.addParticle(m_B)

lj = openmm.NonbondedForce()
lj.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
lj.setCutoffDistance(r_cut)

for i_A in range(N_A):
    lj.addParticle(0.0, sigma_A, epsilon_A)
for i_B in range(N_B):
    lj.addParticle(0.0, sigma_B, epsilon_B)

system.addForce(lj)

system.addForce(openmm.MonteCarloBarostat(press, temp))
integrator = openmm.LangevinMiddleIntegrator(temp, friction, step)

# Define the Topology

topology = openmm.app.Topology()

chain_A = topology.addChain()
chain_B = topology.addChain()

for i_A in range(N_A):
    residue = topology.addResidue("A", chain_A)
    topology.addAtom("A", None, residue)
for i_B in range(N_B):
    residue = topology.addResidue("B", chain_B)
    topology.addAtom("B", None, residue)

# Prepare the Simulation

simulation = openmm.app.Simulation(topology, system, integrator)

grid_a, grid_b = [], []

for i_x in range(10):
    for i_y in range(10):
        for i_z in range(10):
            (grid_a, grid_b)[(i_x + i_y + i_z) % 2].append(
                openmm.Vec3(i_x / 10, i_y / 10, i_z / 10) * L_box
            )

simulation.context.setPositions(grid_a + grid_b)

# Minimize

print("Performing energy minimization...")
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temp)

# Simulate

print("Simulating...")
dcdReporter = openmm.app.DCDReporter("trajectory.dcd", 100)
simulation.reporters.append(dcdReporter)
simulation.step(10000)
