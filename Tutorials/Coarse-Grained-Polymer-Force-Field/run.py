from sys import stdout

import numpy as np
import openmm as mm
from openmm import app, unit

# Define Topology

cgElement = app.Element(number=1000, name="CG-element", symbol="CG", mass=120)

topology = app.Topology()

M = 100  # number of polymer chains
N = 10  # number of atoms in each chain

for m in range(M):
    chain = topology.addChain()
    residue = topology.addResidue(name="CG-residue", chain=chain)
    atom1 = topology.addAtom(name="CG-bead", element=cgElement, residue=residue)
    for i in range(1, N):
        residue = topology.addResidue(name="CG-residue", chain=chain)
        atom2 = topology.addAtom(name="CG-bead", element=cgElement, residue=residue)
        topology.addBond(atom1, atom2)
        atom1 = atom2

print(topology)

# Define Initial Positions

positions = []
for m in range(M):
    x0 = np.array(((m % 10) * 1.0, (m // 10) * 1.0, 0))
    positions.append(x0)
    for i in range(1, N):
        xi = positions[-1] + np.array((0, 0, 0.38))
        positions.append(xi)

positions = positions * unit.nanometer
assert len(positions) == topology.getNumAtoms()

topology.setPeriodicBoxVectors(np.eye(3) * 11.0 * unit.nanometers)

with open("initial_config.pdb", "w") as f:
    app.PDBFile.writeFile(topology, positions, f)


# Build System

system = mm.System()
system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
for atom in topology.atoms():
    system.addParticle(atom.element.mass)

harmonic_bond_force = mm.HarmonicBondForce()

for bond in topology.bonds():
    harmonic_bond_force.addBond(bond.atom1.index, bond.atom2.index, 0.38, 1000)

expression = (
    "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
    + " sigma=0.5*(sigma1+sigma2);"
    + " epsilon=sqrt(epsilon1*epsilon2)"
)
custom_nb_force = mm.CustomNonbondedForce(expression)
custom_nb_force.addPerParticleParameter("sigma")
custom_nb_force.addPerParticleParameter("epsilon")
for atom in topology.atoms():
    custom_nb_force.addParticle([0.5, 1.0])

custom_nb_force.createExclusionsFromBonds(
    [(bond[0].index, bond[1].index) for bond in topology.bonds()], 1
)
custom_nb_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
custom_nb_force.setCutoffDistance(1.5 * unit.nanometers)

system.addForce(harmonic_bond_force)
system.addForce(custom_nb_force)

with open("system.xml", "w") as output:
    output.write(mm.XmlSerializer.serialize(system))

# Prepare the Simulation

integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 0.01 / unit.picosecond, 0.010 * unit.picoseconds
)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

simulation.reporters.append(app.DCDReporter("traj.dcd", 1000, enforcePeriodicBox=False))
simulation.reporters.append(
    app.StateDataReporter(
        stdout,
        10000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
        speed=True,
    )
)

# Equilibrate
simulation.step(10000)

barostatIndex = system.addForce(
    mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin)
)
simulation.context.reinitialize(preserveState=True)
simulation.step(100000)

with open("equilibrated_config.pdb", "w") as f:
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    app.PDBFile.writeFile(topology, state.getPositions(), f)
