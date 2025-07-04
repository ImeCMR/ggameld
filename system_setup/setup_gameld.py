#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meld
from meld import unit as u
from meld import parse
from meld.system import montecarlo as mc
from meld.system.scalers import LinearRamp
from meld import system

N_REPLICAS = 10
N_STEPS = 133333                # 500 ns (total across all replicas)
BLOCK_SIZE = 500
TIMESTEPS = 2500                # 5 ps
CONVENTIONAL_MD_PREP = 100      # 0.5 ns
CONVENTIONAL_MD = 900           # 4.5 ns
GAMD_EQUILIBRATION_PREP = 900   # 4.5 ns
GAMD_EQUILIBRATION = 8100       # 40.5 ns

hydrophobes_res = ['ALA','ILE','LEU','MET','PHE','PRO','TRP','VAL']

def gen_state(s, index):
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state

def make_ss_groups(subset=None):
    active = 0
    extended = 0
    sse = []
    ss = open('ss.dat','r').readlines()[0]
    for i,l in enumerate(ss.rstrip()):
        if l not in "HE.":
            continue
        if l not in 'E' and extended:
            end = i
            sse.append((start+1,end))
            extended = 0
        if l in 'E':
            if i+1 in subset:
                active += 1
            if extended:
                continue
            else:
                start = i
                extended = 1
    return sse,active

def create_hydrophobes(s, group_1=np.array([]), group_2=np.array([]), CO=True):
    hy_rest=open('hydrophobe.dat','w')
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

    sequence = [(i,j) for i,j in zip(s.residue_numbers,s.residue_names)]
    sequence = sorted(set(sequence))
    sequence = dict(sequence)

    group_1 = [ res for res in group_1 if (sequence[res-1] in hydrophobes_res) ]
    group_2 = [ res for res in group_2 if (sequence[res-1] in hydrophobes_res) ]

    pairs = []
    for i in group_1:
        for j in group_2:
            if ( (i,j) in pairs ) or ( (j,i) in pairs ):
                continue
            if ( i == j ):
                continue
            if (abs(i-j)< 7):
                continue
            pairs.append( (i,j) )
            atoms_i = atoms[sequence[i-1]]
            atoms_j = atoms[sequence[j-1]]
            for a_i in atoms_i:
                for a_j in atoms_j:
                    hy_rest.write('{} {} {} {}\n'.format(i,a_i, j, a_j))
            hy_rest.write('\n')

def generate_strand_pairs(s,sse,subset=np.array([]),CO=True):
    f=open('strand_pair.dat','w')
    n_res = s.residue_numbers[-1]
    subset = subset if subset.size else np.array(list(range(n_res)))+1
    for i in range(len(sse)):
        start_i,end_i = sse[i]
        for j in range(i+1,len(sse)):
            start_j,end_j = sse[j]
            for res_i in range(start_i,end_i+1):
                for res_j in range(start_j,end_j+1):
                    if res_i in subset or res_j in subset:
                        f.write('{} {} {} {}\n'.format(res_i, 'N', res_j, 'O'))
                        f.write('{} {} {} {}\n'.format(res_i, 'O', res_j, 'N'))
                        f.write('\n')

def get_dist_restraints_hydrophobe(filename, s, scaler, ramp, seq):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=0.0*u.nanometer, r3=0.5*u.nanometer, r4=0.7*u.nanometer,
                                                 k=250*u.kilojoule_per_mole/u.nanometer **2,
                                                 atom1=s.index.atom(i,name_i, expected_resname=seq[i][-3:]),
                                                 atom2=s.index.atom(j,name_j, expected_resname=seq[j][-3:]))
            rest_group.append(rest)
    return dists

def get_dist_restraints_strand_pair(filename, s, scaler, ramp, seq):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=0.0*u.nanometer, r3=0.35*u.nanometer, r4=0.55*u.nanometer,
                                                 k=250*u.kilojoule_per_mole/u.nanometer **2,
                                                 atom1=s.index.atom(i,name_i, expected_resname=seq[i][-3:]),
                                                 atom2=s.index.atom(j,name_j, expected_resname=seq[j][-3:]))
            rest_group.append(rest)
    return dists


def setup_system():

    # Load the sequence from file
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())

    # Build the system using minimized pdb to start
    p = meld.AmberSubSystemFromPdbFile('1DV0_min.pdb')
    build_options = meld.AmberOptions(
        forcefield="ff14sbside",
        implicit_solvent_model='gbNeck2',
        use_big_timestep=True,
        cutoff=1.8*u.nanometer,
        remove_com=False,
        enable_amap=False,
        amap_beta_bias=1.0,
        enable_gamd=True,  # Enable GaMD
        boost_type_str="upper-total",
        conventional_md_prep=CONVENTIONAL_MD_PREP * TIMESTEPS,
        conventional_md=CONVENTIONAL_MD * TIMESTEPS,
        gamd_equilibration_prep=GAMD_EQUILIBRATION_PREP * TIMESTEPS,
        gamd_equilibration=GAMD_EQUILIBRATION * TIMESTEPS,
        total_simulation_length=N_STEPS * TIMESTEPS,
        averaging_window_interval=TIMESTEPS,
        sigma0p=6.0,
        sigma0d=6.0,
        random_seed=0,
        friction_coefficient=1.0,
    )

    builder = meld.AmberSystemBuilder(build_options)
    s = builder.build_system([p]).finalize()


    # temperature ladder across all replicas from 1 - num_replicas
    s.temperature_scaler = system.temperature.GeometricTemperatureScaler(0, 1, 300.*u.kelvin, 550.*u.kelvin)

    ramp = LinearRamp(0, TIMESTEPS, 0, 1)

    seq = sequence.split()
    for i in range(len(seq)):
        if seq[i][-3:] == 'HIE':
            seq[i] = 'HIS'

    # Secondary structure restraints
    ss_scaler = s.restraints.create_scaler('constant')
    ss_rests = parse.get_secondary_structure_restraints(
        filename='ss.dat',
        system=s,
        scaler=ss_scaler,
        ramp=ramp,
        torsion_force_constant=0.01*u.kilojoule_per_mole/u.degree**2,
        distance_force_constant=2.5*u.kilojoule_per_mole/u.nanometer**2,
        quadratic_cut=2.0*u.nanometer
    )
    n_ss_keep = int(len(ss_rests) * 0.85)
    s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)

    # Confinement restraints
    conf_scaler = s.restraints.create_scaler('constant')
    confinement_rests = []
    for index in range(n_res):
        rest = s.restraints.create_restraint(
            'confine',
            conf_scaler,
            ramp=ramp,
            atom_index=s.index.atom(index, 'CA', expected_resname=seq[index][-3:]),
            radius=4.5*u.nanometer,
            force_const=250.0*u.kilojoule_per_mole/u.nanometer**2
        )
        confinement_rests.append(rest)
    s.restraints.add_as_always_active_list(confinement_rests)

    # Distance restraints setup
    scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    subset1 = np.array(list(range(n_res))) + 1

    # Hydrophobic contacts
    create_hydrophobes(s, group_1=subset1, group_2=subset1, CO=False)

    dists = get_dist_restraints_hydrophobe('hydrophobe.dat', s, scaler, ramp, seq)
    s.restraints.add_selectively_active_collection(dists, int(1.2 * len([r for r in seq if r in hydrophobes_res])))

    # Strand pairing
    #sse, active = make_ss_groups(subset=subset1)
    #generate_strand_pairs(s, sse, subset=subset1, CO=False)

    #dists = get_dist_restraints_strand_pair('strand_pair.dat', s, scaler, ramp, seq)
    #s.restraints.add_selectively_active_collection(dists, int(0.45 * active))

    # Setup MCMC movers like in normal MELD script
    movers = []
    n_atoms = s.n_atoms
    for i in range(n_res):
        n = s.index.atom(i, 'N', expected_resname=seq[i][-3:])
        ca = s.index.atom(i, 'CA', expected_resname=seq[i][-3:])
        c = s.index.atom(i, 'C', expected_resname=seq[i][-3:])
        mover = mc.DoubleTorsionMover(
            index1a=n,
            index1b=ca,
            atom_indices1=[system.indexing.AtomIndex(j) for j in range(ca, n_atoms)],
            index2a=ca,
            index2b=c,
            atom_indices2=[system.indexing.AtomIndex(j) for j in range(c, n_atoms)]
        )
        movers.append((mover, 1))

    sched = mc.MonteCarloScheduler(movers, n_res * 60)

    # Create MELD RunOptions including MC scheduler
    options = meld.RunOptions(
        timesteps=TIMESTEPS,
        minimize_steps=20000,
        min_mc=sched,
        param_mcmc_steps=200,
        enable_gamd=True  # Enable GaMD
    )

    # Setup REMD runner with GaMD enabled
    remd = meld.setup_replica_exchange(
        s,
        n_replicas=N_REPLICAS,
        n_steps=N_STEPS,
        n_trials=48*48,
        adaptation_growth_factor=2.0,
        adaptation_burn_in=50,
        adaptation_adapt_every=50
    )

    # Setup data store that creates all required files
    meld.setup_data_store(s, options, remd, block_size=BLOCK_SIZE)

    return s.n_atoms

setup_system()
