import logging
import sys

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmml import MLPotential

logging.basicConfig(level=logging.ERROR)

# Input File

pdb = PDBFile("alanine-dipeptide.pdb")
print(pdb.topology)

# Create System

potential = MLPotential("aceff-2.0")
system = potential.createSystem(pdb.topology)

# Prepare a Simulation

integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.001 * picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300 * kelvin)

# Equilibrate

simulation.step(100)

# Simulate

simulation.reporters.append(XTCReporter("alanine-dipeptide.xtc", 100))
simulation.reporters.append(
    StateDataReporter(
        sys.stdout, 100, potentialEnergy=True, temperature=True, step=True, speed=True
    )
)
simulation.step(10000)

# Plot

u = mda.Universe("alanine-dipeptide.pdb", "alanine-dipeptide.xtc")
ramachandran = Ramachandran(u.select_atoms("protein")).run()
ramachandran.plot()
