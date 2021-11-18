import tequila as tq
import torch
import numpy as np
from matplotlib import pyplot as plt

bond_lengths=np.linspace(.3,1.6,20) # our bond length, in angstrom.
amp_arrays = []
state_preps = []
for i in bond_lengths:
    # the line below initializes a tequila molecule object for H2 at a specific bond length.
    # see the quantum chemistry tutorial for more details.
    molecule = tq.chemistry.Molecule(geometry = "H 0.0 0.0 0.0\n H 0.0 0.0 {}".format(str(i)), basis_set="sto-3g")
    amplitude = molecule.compute_amplitudes(method='ccsd') # get the state prep amplitudes
    amp_arrays.append(np.asarray([v for v in amplitude.make_parameter_dictionary().values()]))
    state_preps.append(molecule.make_uccsd_ansatz(trotter_steps=1,initial_amplitudes=amplitude))