import matplotlib.pyplot as plt
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# Input Files

pdbx = PDBxFile("1AKI-processed.cif")
forcefield = ForceField("amber19-all.xml", "amber19/tip3pfb.xml")

# System Configuration

nonbondedMethod = PME
nonbondedCutoff = 1.0 * nanometers
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 1.5 * amu
ewaldErrorTolerance = 0.0005

# Integration Options

dt = 0.004 * picoseconds
temperature = 300 * kelvin
friction = 1.0 / picosecond
pressure = 1.0 * atmospheres
barostatInterval = 25

# Simulation Options

steps = 20000
equilibrationSteps = 1000
dataReporter = StateDataReporter(
    "log.txt",
    100,
    totalSteps=steps,
    step=True,
    speed=True,
    progress=True,
    potentialEnergy=True,
    temperature=True,
    density=True,
    separator="\t",
)

# Prepare the Simulation

print("Building system...")
topology = pdbx.topology
positions = pdbx.positions
system = forcefield.createSystem(
    topology,
    nonbondedMethod=nonbondedMethod,
    nonbondedCutoff=nonbondedCutoff,
    constraints=constraints,
    rigidWater=rigidWater,
    ewaldErrorTolerance=ewaldErrorTolerance,
    hydrogenMass=hydrogenMass,
)
system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Minimize and Equilibrate

print("Performing energy minimization...")
simulation.minimizeEnergy()
print("Equilibrating...")
simulation.context.setVelocitiesToTemperature(temperature)
simulation.step(equilibrationSteps)

# Simulate

print("Simulating...")
simulation.reporters.append(dataReporter)
simulation.currentStep = 0
simulation.step(steps)

# Plot Data

print("Plotting...")
step, pe, temp, rho = np.loadtxt("log.txt", delimiter="\t", usecols=(1, 2, 3, 4)).T
time = step * dt.value_in_unit(picosecond)

fig, ax = plt.subplots(layout="constrained")
ax.plot(time, pe)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Potential Energy (kJ/mol)")
fig.savefig("pe.png")

fig, ax = plt.subplots(layout="constrained")
ax.plot(time, temp)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Temperature (K)")
fig.savefig("temp.png")

fig, ax = plt.subplots(layout="constrained")
ax.plot(time, rho)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Density (g/mL)")
fig.savefig("rho.png")
