import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmtools.testsystems import LennardJonesFluid
from pymbar import MBAR, timeseries

# Simulation Parameters

pressure = 80 * atmospheres
temperature = 120 * kelvin
collision_rate = 5 / picoseconds
timestep = 2.5 * femtoseconds

# Create LJ Fluid

sigma = 3.4 * angstrom
epsilon = 0.238 * kilocalories_per_mole
fluid = LennardJonesFluid(sigma=sigma, epsilon=epsilon)
[topology, system, positions] = [fluid.topology, fluid.system, fluid.positions]

# Add Monte Carlo Barostat

barostat = MonteCarloBarostat(pressure, temperature)
system.addForce(barostat)

# Create Custom Alchemical Force

(nbforce,) = (
    force for force in system.getForces() if isinstance(force, NonbondedForce)
)

alchemical_particles = {0}
chemical_particles = set(range(system.getNumParticles())) - alchemical_particles

energy_function = "lambda*4*epsilon*x*(x-1.0); x = (sigma/reff_sterics)^6;"
energy_function += "reff_sterics = sigma*(0.5*(1.0-lambda) + (r/sigma)^6)^(1/6);"
energy_function += "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2);"
custom_force = CustomNonbondedForce(energy_function)
custom_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
custom_force.setCutoffDistance(nbforce.getCutoffDistance())
custom_force.addGlobalParameter("lambda", 1.0)

custom_force.addPerParticleParameter("sigma")
custom_force.addPerParticleParameter("epsilon")
for index in range(system.getNumParticles()):
    [charge, sigma, epsilon] = nbforce.getParticleParameters(index)
    custom_force.addParticle([sigma, epsilon])
    if index in alchemical_particles:
        # remove the alchemical particle from the existing NonBondedForce
        nbforce.setParticleParameters(index, charge * 0, sigma, epsilon * 0)

# custom force occurs between alchemical particle and the other particles
custom_force.addInteractionGroup(alchemical_particles, chemical_particles)
system.addForce(custom_force)


# Prepare the Simulation

integrator = LangevinIntegrator(temperature, collision_rate, timestep)
simulation = Simulation(topology, system, integrator)
simulation.context.setPositions(positions)

# Minimize
print("Performing energy minimization...")
simulation.minimizeEnergy()

# Perform Alchemical Simulations

nsteps = 2500  # steps per sample
niterations = 500  # samples

lambdas = np.linspace(1.0, 0.0, 10)  # alchemical lambda schedule
nstates = len(lambdas)
u_kln = np.zeros([nstates, nstates, niterations], np.float64)
kT = AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB * integrator.getTemperature()
for k in range(nstates):
    for iteration in range(niterations):
        print(f"state {k} iteration {iteration} / {niterations}")
        # set alchemical state
        simulation.context.setParameter("lambda", lambdas[k])
        # run dynamics
        simulation.step(nsteps)
        # compute energies at all alchemical states
        for l in range(nstates):
            simulation.context.setParameter("lambda", lambdas[l])
            u_kln[k, l, iteration] = (
                simulation.context.getState(getEnergy=True).getPotentialEnergy() / kT
            )


# Estimate Free Energy

N_k = np.zeros([nstates], np.int32)  # number of uncorrelated samples
for k in range(nstates):
    [nequil, g, Neff_max] = timeseries.detect_equilibration(u_kln[k, k, :])
    indices = timeseries.subsample_correlated_data(u_kln[k, k, :], g=g)
    N_k[k] = len(indices)
    u_kln[k, :, 0 : N_k[k]] = u_kln[k, :, indices].T

mbar = MBAR(u_kln, N_k)
results = mbar.compute_free_energy_differences(compute_uncertainty=True)

print(
    f"Free Energy: {results['Delta_f'][nstates - 1, 0]} +- {results['dDelta_f'][nstates - 1, 0]}"
)
