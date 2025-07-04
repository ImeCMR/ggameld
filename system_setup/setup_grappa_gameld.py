#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meld
import meld.system as system
from meld.remd import ladder, adaptor, leader
import meld.system.montecarlo as mc
from meld import comm, vault
from meld import parse
from meld import remd
from meld.system import param_sampling, scalers
from openmm import unit
from openmm.app import PDBFile, Modeller, ForceField
from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder

# Script parameters (can be adjusted by user)
N_REPLICAS = 10
N_STEPS = 133333
BLOCK_SIZE = 500

# GaMD specific timing parameters (number of blocks of TIMESTEPS)
TIMESTEPS = 2500
CONVENTIONAL_MD_PREP = 100
CONVENTIONAL_MD = 900
GAMD_EQUILIBRATION_PREP = 900
GAMD_EQUILIBRATION = 8100
TOTAL_GAMD_SIM_STEPS = N_STEPS * TIMESTEPS

# Other GaMD parameters (can be adjusted)
DEFAULT_BOOST_TYPE = "upper-total"
DEFAULT_SIGMA0P = 6.0
DEFAULT_SIGMA0D = 6.0
GAMD_RANDOM_SEED = 0
DEFAULT_FRICTION_COEFFICIENT = 1.0
DEFAULT_TEMPERATURE_K = 300.0
USE_BIG_TIMESTEP_GAMD = False


hydrophobes = 'AILMFPWV'
hydrophobes_res = ['ALA','ILE','LEU','MET','PHE','PRO','TRP','VAL']


def gen_state(s, index):
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state

def make_ss_groups(subset=None):
    active = 0
    extended = 0
    sse = []
    # Ensure ss.dat exists or handle FileNotFoundError
    try:
        with open('ss.dat', 'r') as f:
            ss_line = f.readlines()[0].strip()
    except FileNotFoundError:
        print("Warning: ss.dat not found. No secondary structure elements will be processed for strand pairing.")
        return [], 0

    for i, l in enumerate(ss_line):
        if l not in "HE.":
            continue
        if l not in 'E' and extended:
            end = i
            sse.append((start + 1, end))
            extended = 0
        if l in 'E':
            if subset is None or (i + 1) in subset: # Ensure subset check is safe
                active = active + 1
            if extended:
                continue
            else:
                start = i
                extended = 1
    if extended: # Check if the last element was extended
        sse.append((start + 1, len(ss_line))) # Use length of ss_line for the end of last segment

    print(active, ':number of E residues in subset (or total if no subset)')
    print(sse, ':E residue ranges')
    return sse, active

def create_hydrophobes(s,group_1=np.array([]),group_2=np.array([]),CO=True):
    with open('hydrophobe.dat', 'w') as hy_rest:
        atoms = {"ALA":['CA','CB'],
                 "VAL":['CA','CB','CG1','CG2'],
                 "LEU":['CA','CB','CG','CD1','CD2'],
                 "ILE":['CA','CB','CG1','CG2','CD1'],
                 "PHE":['CA','CB','CG','CD1','CE1','CZ','CE2','CD2'],
                 "TRP":['CA','CB','CG','CD1','NE1','CE2','CZ2','CH2','CZ3','CE3','CD2'],
                 "MET":['CA','CB','CG','SD','CE'],
                 "PRO":['CD','CG','CB','CA']}
        n_res = s.residue_numbers[-1]
        group_1 = group_1 if group_1.size else np.array(list(range(n_res)))+1
        group_2 = group_2 if group_2.size else np.array(list(range(n_res)))+1

        sequence_dict = {res_num: name for res_num, name in zip(s.residue_numbers, s.residue_names)}
        # Use a dictionary for faster lookups if residue_numbers are not contiguous or 0-indexed
        # For this script, assuming residue_numbers from PDB are 1-indexed and mostly contiguous.
        # The provided sequence from System object might be more reliable if PDB parsing is complex.
        # Let's assume s.residue_names and s.residue_numbers are aligned and 0-indexed from MELD's perspective

        # Correctly get unique residue info for mapping
        # MELD system residue_numbers are 0-indexed, residue_names are per-atom
        unique_res_info = sorted(list(set(zip(s.residue_numbers, s.residue_names))))
        # seq_map = {res_idx: res_name for res_idx, res_name in unique_res_info} # 0-indexed
        # For 1-indexed PDB numbers used in .dat files:
        seq_map_1_indexed = {res_idx + 1: res_name for res_idx, res_name in unique_res_info}


        group_1_filtered = [res for res in group_1 if seq_map_1_indexed.get(res, "") in hydrophobes_res]
        group_2_filtered = [res for res in group_2 if seq_map_1_indexed.get(res, "") in hydrophobes_res]

        pairs = []
        for i in group_1_filtered:
            for j in group_2_filtered:
                if ( (i,j) in pairs ) or ( (j,i) in pairs ):
                    continue
                if ( i ==j ):
                    continue
                if (abs(i-j)< 7): # sequence separation
                    continue
                pairs.append( (i,j) )

                res_i_name = seq_map_1_indexed.get(i)
                res_j_name = seq_map_1_indexed.get(j)

                if not res_i_name or not res_j_name: continue # Should not happen if filtered correctly

                atoms_i = atoms[res_i_name]
                atoms_j = atoms[res_j_name]

                for a_i in atoms_i:
                    for a_j in atoms_j:
                        hy_rest.write('{} {} {} {}\n'.format(i,a_i, j, a_j))
                hy_rest.write('\n')

def generate_strand_pairs(s,sse,subset=np.array([]),CO=True):
    with open('strand_pair.dat', 'w') as f:
        n_res = s.residue_numbers[-1] # This is max residue index (0-based)
        # subset is 1-indexed from PDB numbers
        # sse is also 1-indexed from ss.dat

        for i in range(len(sse)):
            start_i,end_i = sse[i]
            for j in range(i+1,len(sse)):
                start_j,end_j = sse[j]
                for res_i in range(start_i,end_i+1):
                    for res_j in range(start_j,end_j+1):
                        # Check if subset is relevant or just use all
                        in_subset_i = subset is None or res_i in subset
                        in_subset_j = subset is None or res_j in subset
                        if in_subset_i or in_subset_j: # if either is in subset (or no subset defined)
                            f.write('{} {} {} {}\n'.format(res_i, 'N', res_j, 'O'))
                            f.write('{} {} {} {}\n'.format(res_i, 'O', res_j, 'N'))
                            f.write('\n')

def get_dist_restraints_hydrophobe(filename, s, scaler, ramp, seq_map_0_indexed_tuple):
    dists = []
    rest_group = []
    try:
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. No hydrophobic restraints will be added.")
        return []

    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            if rest_group: # only add if group is not empty
                dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            res_i_1_idx = int(cols[0])
            name_i = cols[1]
            res_j_1_idx = int(cols[2])
            name_j = cols[3]

            # seq_map_0_indexed_tuple is like [(0, 'ALA'), (1, 'ARG'), ...]
            # We need to map 1-indexed PDB resnum to 0-indexed MELD resnum for s.index.atom
            # And then get the expected resname for that 0-indexed MELD resnum.
            try:
                atom1_res_idx_0 = res_i_1_idx - 1 # Convert PDB 1-idx to MELD 0-idx
                atom2_res_idx_0 = res_j_1_idx - 1 # Convert PDB 1-idx to MELD 0-idx

                # Find expected resname from our seq_map_0_indexed_tuple for validation
                expected_resname_i = next(name for idx, name in seq_map_0_indexed_tuple if idx == atom1_res_idx_0)
                expected_resname_j = next(name for idx, name in seq_map_0_indexed_tuple if idx == atom2_res_idx_0)

                rest = s.restraints.create_restraint(
                    'distance', scaler, ramp,
                    r1=0.0 * unit.nanometer, r2=0.0 * unit.nanometer,
                    r3=0.5 * unit.nanometer, r4=0.7 * unit.nanometer,
                    k=250 * unit.kilojoule_per_mole / unit.nanometer ** 2,
                    atom1=s.index.atom(atom1_res_idx_0, name_i, expected_resname=expected_resname_i),
                    atom2=s.index.atom(atom2_res_idx_0, name_j, expected_resname=expected_resname_j)
                )
                rest_group.append(rest)
            except StopIteration:
                print(f"Warning: Could not find residue {res_i_1_idx} or {res_j_1_idx} in sequence map for restraint: {line}")
            except Exception as e:
                print(f"Warning: Error creating restraint for line '{line}': {e}")

    if rest_group: # Add any remaining restraints
        dists.append(s.restraints.create_restraint_group(rest_group, 1))
    return dists

def get_dist_restraints_strand_pair(filename, s, scaler, ramp, seq_map_0_indexed_tuple):
    dists = []
    rest_group = []
    try:
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. No strand_pair restraints will be added.")
        return []

    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            if rest_group:
                dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            res_i_1_idx = int(cols[0])
            name_i = cols[1]
            res_j_1_idx = int(cols[2])
            name_j = cols[3]

            try:
                atom1_res_idx_0 = res_i_1_idx - 1
                atom2_res_idx_0 = res_j_1_idx - 1
                expected_resname_i = next(name for idx, name in seq_map_0_indexed_tuple if idx == atom1_res_idx_0)
                expected_resname_j = next(name for idx, name in seq_map_0_indexed_tuple if idx == atom2_res_idx_0)

                rest = s.restraints.create_restraint(
                    'distance', scaler, ramp,
                    r1=0.0 * unit.nanometer, r2=0.0 * unit.nanometer,
                    r3=0.35 * unit.nanometer, r4=0.55 * unit.nanometer,
                    k=250 * unit.kilojoule_per_mole / unit.nanometer ** 2,
                    atom1=s.index.atom(atom1_res_idx_0, name_i, expected_resname=expected_resname_i),
                    atom2=s.index.atom(atom2_res_idx_0, name_j, expected_resname=expected_resname_j)
                )
                rest_group.append(rest)
            except StopIteration:
                print(f"Warning: Could not find residue {res_i_1_idx} or {res_j_1_idx} in sequence map for restraint: {line}")
            except Exception as e:
                print(f"Warning: Error creating restraint for line '{line}': {e}")

    if rest_group:
        dists.append(s.restraints.create_restraint_group(rest_group, 1))
    return dists


#######################################

def setup_system():

    sequence_data = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence_data.split()) # n_res is count of residues

    pdb_file = PDBFile('protein_min.pdb')
    forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')
    modeller = Modeller(pdb_file.topology, pdb_file.positions)
    modeller.addHydrogens(forcefield)
    topology = modeller.topology
    positions = modeller.positions

    grappa_options = GrappaOptions(
        solvation_type="implicit",
        grappa_model_tag="grappa-1.4.0",
        base_forcefield_files=['amber14/protein.ff14SB.xml', 'implicit/gbn2.xml'],
        default_temperature=DEFAULT_TEMPERATURE_K * unit.kelvin,
        cutoff=None,
        use_big_timestep=USE_BIG_TIMESTEP_GAMD,
        remove_com=True,
        enable_gamd=True,
        boost_type_str=DEFAULT_BOOST_TYPE,
        conventional_md_prep=CONVENTIONAL_MD_PREP * TIMESTEPS,
        conventional_md=CONVENTIONAL_MD * TIMESTEPS,
        gamd_equilibration_prep=GAMD_EQUILIBRATION_PREP * TIMESTEPS,
        gamd_equilibration=GAMD_EQUILIBRATION * TIMESTEPS,
        total_simulation_length=TOTAL_GAMD_SIM_STEPS,
        averaging_window_interval=TIMESTEPS,
        sigma0p=DEFAULT_SIGMA0P,
        sigma0d=DEFAULT_SIGMA0D,
        gamd_random_seed=GAMD_RANDOM_SEED,
        gamd_friction_coefficient=DEFAULT_FRICTION_COEFFICIENT,
    )

    grappa_builder = GrappaSystemBuilder(grappa_options)
    system_spec = grappa_builder.build_system(topology, positions)
    s = system_spec.finalize()

    s.temperature_scaler = system.temperature.GeometricTemperatureScaler(
        alpha_min=0.0, alpha_max=1.0,
        temperature_min=DEFAULT_TEMPERATURE_K * unit.kelvin,
        temperature_max=DEFAULT_TEMPERATURE_K * 1.83 * unit.kelvin
    )

    ramp = scalers.LinearRamp(time_min=0, time_max=TIMESTEPS, value_min=0.0, value_max=1.0)

    # Create a 0-indexed sequence map (tuple of tuples for easy iteration)
    # s.residue_names and s.residue_numbers are per-atom. Need unique per-residue.
    # This creates a list of (0-indexed res_number, RESNAME)
    unique_residue_info_0_indexed = tuple(sorted(list(set(zip(s.residue_numbers, s.residue_names)))))


    hydrophobic_res_in_protein_count = sum(1 for _,name in unique_residue_info_0_indexed if name in hydrophobes_res)
    print(hydrophobic_res_in_protein_count,':number of hydrophobic residue')

    ss_scaler = s.restraints.create_scaler('constant')
    # For get_secondary_structure_restraints, 'seq' arg is not used if system is provided.
    # Internally it uses s.index.atom which needs expected_resname based on MELD's 0-indexed residue numbers.
    ss_rests = parse.get_secondary_structure_restraints(filename='ss.dat', system=s, scaler=ss_scaler,
            ramp=ramp, torsion_force_constant=0.01*unit.kilojoule_per_mole/unit.degree **2,
            distance_force_constant=2.5*unit.kilojoule_per_mole/unit.nanometer **2, quadratic_cut=2.0*unit.nanometer)
    if ss_rests:
        n_ss_keep = int(len(ss_rests) * 0.85)
        s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)

    conf_scaler = s.restraints.create_scaler('constant')
    confinement_rests = []
    for res_idx, res_name_for_conf in unique_residue_info_0_indexed: # Iterate over unique 0-indexed residues
        rest = s.restraints.create_restraint('confine', conf_scaler, ramp=ramp,
                                             atom_index=s.index.atom(res_idx, 'CA', expected_resname=res_name_for_conf),
                                             radius=4.5*unit.nanometer, force_const=250.0*unit.kilojoule_per_mole/unit.nanometer **2)
        confinement_rests.append(rest)
    if confinement_rests:
        s.restraints.add_as_always_active_list(confinement_rests)

    dist_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    subset1_1_indexed = np.array(list(range(n_res))) + 1 # 1-indexed for PDB numbers

    create_hydrophobes(s,group_1=subset1_1_indexed,group_2=subset1_1_indexed,CO=False)
    hydrophobe_dists = get_dist_restraints_hydrophobe('hydrophobe.dat', s, dist_scaler, ramp, unique_residue_info_0_indexed)
    if hydrophobe_dists:
      s.restraints.add_selectively_active_collection(hydrophobe_dists, int(1.2 * hydrophobic_res_in_protein_count))

    sse_1_indexed, active_strand_res_count = make_ss_groups(subset=subset1_1_indexed)
    generate_strand_pairs(s,sse_1_indexed,subset=subset1_1_indexed,CO=False)
    strand_pair_dists = get_dist_restraints_strand_pair('strand_pair.dat', s, dist_scaler, ramp, unique_residue_info_0_indexed)
    if active_strand_res_count > 0 and strand_pair_dists:
        s.restraints.add_selectively_active_collection(strand_pair_dists, int(0.45 * active_strand_res_count))
    else:
        print("No strand pairing restraints added as no active strand residues or no strand pair definitions found.")

    movers = []
    n_atoms_total = s.n_atoms
    for res_idx, res_name_for_mcmc in unique_residue_info_0_indexed:
        try:
            n_atom_idx = s.index.atom(res_idx, 'N', expected_resname=res_name_for_mcmc)
            ca_atom_idx = s.index.atom(res_idx, 'CA', expected_resname=res_name_for_mcmc)
            c_atom_idx = s.index.atom(res_idx, 'C', expected_resname=res_name_for_mcmc)

            # Define atom indices for phi and psi torsions carefully
            # Phi: C(i-1) - N(i) - CA(i) - C(i)
            # Psi: N(i) - CA(i) - C(i) - N(i+1)
            # DoubleTorsionMover acts on backbone atoms of residue i and subsequent atoms

            # Atoms to move with phi torsion (N-CA axis): CA(i) and everything attached to/after CA(i) in sequence
            # This is generally from CA(i) to the end of the chain.
            phi_move_indices = list(system.indexing.AtomIndex(k) for k in range(int(ca_atom_idx), n_atoms_total))

            # Atoms to move with psi torsion (CA-C axis): C(i) and everything attached to/after C(i) in sequence
            # This is generally from C(i) to the end of the chain.
            psi_move_indices = list(system.indexing.AtomIndex(k) for k in range(int(c_atom_idx), n_atoms_total))

            mover = mc.DoubleTorsionMover(index1a=n_atom_idx, index1b=ca_atom_idx, atom_indices1=phi_move_indices,
                                          index2a=ca_atom_idx, index2b=c_atom_idx, atom_indices2=psi_move_indices)
            movers.append((mover, 1))
        except Exception as e: # Catch if specific atoms not found for a residue (e.g. N-term, C-term)
            print(f"Skipping MCMC mover setup for residue {res_idx} ({res_name_for_mcmc}): {e}")


    sched = mc.MonteCarloScheduler(movers, n_res * 60 if movers else 0)


    options = meld.RunOptions(
        timesteps=TIMESTEPS,
        minimize_steps=20000,
        min_mc=sched if movers else None,
        run_mc=sched if movers else None,
        param_mcmc_steps=200,
        enable_gamd=True
    )

    store = vault.DataStore(gen_state(s,0), N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    remd_ladder = ladder.NearestNeighborLadder(n_trials=N_REPLICAS * N_REPLICAS)
    adaptation_policy = adaptor.AdaptationPolicy(
        growth_factor=2.0,
        burn_in=50,
        adapt_every=50
    )
    remd_adaptor = adaptor.EqualAcceptanceAdaptor(
        n_replicas=N_REPLICAS,
        adaptation_policy=adaptation_policy,
        min_acc_prob=0.02
    )
    remd_runner = remd.leader.LeaderReplicaExchangeRunner(
        N_REPLICAS,
        max_steps=N_STEPS,
        ladder=remd_ladder,
        adaptor=remd_adaptor
    )
    store.save_remd_runner(remd_runner)

    communicator = comm.MPICommunicator(s.n_atoms, N_REPLICAS, timeout=600000)
    store.save_communicator(communicator)

    initial_states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(initial_states, 0)

    store.save_data_store()

    print(f"System setup complete. N_atoms: {s.n_atoms}")
    print(f"Grappa+GaMELD system configured with {N_REPLICAS} replicas.")
    print(f"Run will proceed for {N_STEPS} REMD exchange cycles.")
    print(f"Each cycle consists of {TIMESTEPS} MD steps.")
    print(f"GaMD Integrator total simulation length (nstlim): {grappa_options.total_simulation_length} steps.")

    return s.n_atoms


if __name__ == '__main__':
    setup_system()
