import matplotlib.pyplot as plt
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# Input Files

inpcrd = AmberInpcrdFile("input.inpcrd")
prmtop = AmberPrmtopFile("input.prmtop", periodicBoxVectors=inpcrd.boxVectors)

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

steps = 100000
equilibrationSteps = 10000
dcdReporter = DCDReporter("trajectory.dcd", 10000)
dataReporter = StateDataReporter(
    "log.txt",
    1000,
    totalSteps=steps,
    step=True,
    speed=True,
    progress=True,
    potentialEnergy=True,
    temperature=True,
    separator="\t",
)
checkpointReporter = CheckpointReporter("checkpoint.chk", 10000)

# Prepare the Simulation

print("Building system...")
topology = prmtop.topology
positions = inpcrd.positions
system = prmtop.createSystem(
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

# Create Customer Reporter


def compute_com_nm(state, indices, masses):
    positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    return (masses[:, None] * positions[indices]).sum(axis=0) / masses.sum()


class COMReporter:
    def __init__(self, file, reportInterval, indices):
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._indices = np.array(indices)
        self._masses = np.array(
            [
                system.getParticleMass(index).value_in_unit(dalton)
                for index in self._indices
            ]
        )

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        # step remaining before next report
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return {"steps": steps, "include": ["positions"], "periodic": False}

    def report(self, simulation, state):
        x, y, z = compute_com_nm(state, self._indices, self._masses)
        print(f"{x:12.6f} {y:12.6f} {z:12.6f}", file=self._out)
        self._out.flush()  # ensure the new line is written immediately


# Simulate

print("Simulating...")
simulation.reporters.append(dcdReporter)
simulation.reporters.append(dataReporter)
simulation.reporters.append(checkpointReporter)

residues = list(topology.residues())[:3]
indices = [atom.index for residue in residues for atom in residue.atoms()]
simulation.reporters.append(COMReporter("com.txt", 100, indices))

simulation.currentStep = 0
simulation.step(steps)

# Plot CoM

print("Plotting...")
com_pos = np.loadtxt("com.txt")

fig = plt.figure(figsize=(5, 4), layout="constrained")

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(*com_pos.T)
ax.set_aspect("equal")
ax.set_xlabel("$x$ (nm)")
ax.set_ylabel("$y$ (nm)")
ax.set_zlabel("$z$ (nm)")
fig.savefig("com.png")
