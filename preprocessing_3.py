'''
This script transforms raw particle-level data into structured datasets for Anomaly Detection in High Energy Physics. 
The workflow follows four main stages:
1. Jet Clustering: Uses the FastJet to group raw Particle Flow (PF) candidates into physical jets
2. Substructure Analysis: Calculates N-subjettiness variables at both a "global" and "sub-jet" level
3. Kinematic Calculation: Computes the Dijet Invariant Mass (mjj​) and angular separation (Δη)
4. Dataset Partitioning: Splits into four sets, shuffels and compresses the data

The dataset from LHC Olympics 2020 events_anomalydetection.h5 is used.
'''

# --- Imports ---
import h5py    
import numpy as np 
import fastjet
import pandas as pd
import awkward as ak
import vector


# --- Configuration & Setup ---
# Define the Signal Region (SR) based on the dijet invariant mass (mjj)
# Events outside this range are considered the Background Region (BR) and spilt into the four datasets
signal_region = (3150, 3850)
out_names = ['/beegfs/u/bbd1146/daten/events_b_br.h5', '/beegfs/u/bbd1146/daten/events_b_sr.h5', '/beegfs/u/bbd1146/daten/events_s_br.h5', '/beegfs/u/bbd1146/daten/events_s_sr.h5']

# Load the raw event data
fnew = pd.read_hdf("/beegfs/u/bbd1146/events_anomalydetection.h5")
events_combined = fnew 
events_np = np.asarray(events_combined)
n_pf = fnew.shape[0]  

all_particles = events_np[:, :2100].reshape(-1, 700, 3)
idx = np.argsort(all_particles[:, :, 0], axis=1)[:, ::-1]
all_particles_sorted = np.take_along_axis(all_particles, idx[:, :, np.newaxis], axis=1)
events_top20 = all_particles_sorted[:, :20, :]

# Initialize arrays to store Particle Flow (PF) candidates for the leading two jets
# Storing 300 values (100 particles * 3 features: pt, eta, phi)
j1pf = np.zeros((n_pf, 300), np.float16)
j2pf = np.zeros((n_pf, 300), np.float16)

# jet_kinematics stores 14 variables: 
# [mjj, delta_eta, jet1(pt, eta, phi, m), jet2(pt, eta, phi, m), jet3(pt, eta, phi, m)]
jet_kinematics = np.zeros((n_pf, 14), np.float64)

# Register awkward to work with 4-vector logic and use the anti-kt algorithm with a jet radius of R=1.0
vector.register_awkward()
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)

# jettiness_features stores 12 variables:
# [4 global N-subjettiness, 4 sub-jettiness for jet 1, 4 sub-jettiness for jet 2]
jettiness_features = np.zeros((n_pf, 12), np.float32)


# --- Event Processing Loop ---
for i in range(len(events_np)):
    if i % 1000 == 0:
        print(f'> computed {i} events')

    sorted_event = events_top20[i]

    pts   = sorted_event[:20, 0]
    etas  = sorted_event[:20, 1]
    phis  = sorted_event[:20, 2]

    # Create a 4-vector array for clustering
    array = ak.Array(
        {"pt": pts, "eta": etas, "phi": phis, "M": np.zeros(len(pts))},
        with_name="Momentum4D",
    )

    # Cluster particles into jets
    cluster    = fastjet.ClusterSequence(array, jetdef)
    jets       = cluster.inclusive_jets(min_pt=0)
    sort_idx   = ak.argsort(jets.pt, ascending=False) # Sort jets by descending transverse momentum (pt)
    jets, all_consts = jets[sort_idx], cluster.constituents()[sort_idx]
    
    n_jets      = len(jets)
    n_jets_save = min(n_jets, 3)

    # Global Jettiness
    taus_global = cluster.njettiness(njets=[1, 2, 3, 4], R0=1.0)
    jettiness_features[i, 0:4] = taus_global

    # Sub-Jettiness Jet 1
    if n_jets > 0:
        c0 = all_consts[0]
        jet0_consts = ak.Array({
            "pt": c0.pt,
            "eta": c0.eta,
            "phi": c0.phi,
            "M": np.zeros(len(c0.pt))
        }, with_name="Momentum4D")
        cluster_sub0 = fastjet.ClusterSequence(jet0_consts, jetdef)
        taus_sub0 = cluster_sub0.njettiness(njets=[1, 2, 3, 4], R0=1.0)
        jettiness_features[i, 4:8] = taus_sub0

    # Sub-Jettiness Jet 2
    if n_jets > 1:
        c1 = all_consts[1]
        jet1_consts = ak.Array({
            "pt": c1.pt,
            "eta": c1.eta,
            "phi": c1.phi,
            "M": np.zeros(len(c1.pt))
        }, with_name="Momentum4D")
        cluster_sub1 = fastjet.ClusterSequence(jet1_consts, jetdef)
        taus_sub1 = cluster_sub1.njettiness(njets=[1, 2, 3, 4], R0=1.0)
        jettiness_features[i, 8:12] = taus_sub1
    
    # Dijet Invariant Mass (mjj) for the two leading jets
    if n_jets > 1:
        E  = jets[0].energy + jets[1].energy
        px = jets[0].px + jets[1].px
        py = jets[0].py + jets[1].py
        pz = jets[0].pz + jets[1].pz
        jet_kinematics[i, 0] = np.sqrt(max(E**2 - px**2 - py**2 - pz**2, 0.0))

    # Jet kinematics + PF candidates
    for j in range(n_jets_save):
        jet = jets[j]
        jet_kinematics[i, 2+j*4 : 2+(j+1)*4] = (jet.pt, jet.eta, jet.phi, jet.mass)

        if j < 2:
            consts = all_consts[j]
            pf_cands = np.stack([
                ak.to_numpy(consts.pt),
                ak.to_numpy(consts.eta),
                ak.to_numpy(consts.phi)
            ], axis=1).ravel()

            len_pf = min(len(pf_cands), 300)
            if j == 0:
                j1pf[i, :len_pf] = pf_cands[:len_pf]
            else:
                j2pf[i, :len_pf] = pf_cands[:len_pf]

# Calculate absolute difference in pseudorapidity between Jet 1 and Jet 2
jet_kinematics[:, 1] = np.abs(jet_kinematics[:, 3] - jet_kinematics[:, 7])  # calculate Delta eta


# --- Data Grouping & Labeling ---
# Extract the ground truth label (Signal=1, Background=0) from the original dataframe
issignal = np.zeros((n_pf, 1), dtype = np.int8)
issignal[:, 0] = fnew.iloc[:, 2100]

groups = []

groups.append(((jet_kinematics[:, 0] < signal_region[0]) | (jet_kinematics[:, 0] > signal_region[1])) & (issignal.reshape(-1) == 0))  # B in BR
groups.append((jet_kinematics[:, 0] >= signal_region[0]) & (jet_kinematics[:, 0] <= signal_region[1]) & (issignal.reshape(-1) == 0))  # B in SR

groups.append(((jet_kinematics[:, 0] < signal_region[0]) | (jet_kinematics[:, 0] > signal_region[1])) & issignal.reshape(-1))  # S in BR
groups.append((jet_kinematics[:, 0] >= signal_region[0]) & (jet_kinematics[:, 0] <= signal_region[1]) & issignal.reshape(-1))  # S in SR

# Sanity check: Ensure every event is assigned to exactly one group
groups = np.array(groups, dtype = np.bool)
if ((groups.sum(0) != 1).sum()) != 0:
    raise AssertionError('Some elements are in no or more than one group')


# --- Shuffling & File Writing ---
n_ev_gr = groups.sum(1)  # number of events in each group
orders = []

# Generate and save random shuffle orders for reproducibility
for i in range(len(groups)):
    order = np.arange(n_ev_gr[i])
    np.random.shuffle(order)
    orders.append(order)
    with open('/beegfs/u/bbd1146/daten/shuffle_order_region_' + str(i) + '.npy', 'wb') as f_order:
        np.save(f_order, order)

# Create HDF5 files for each category
out_files = []
for name in out_names:
    out_files.append(h5py.File(name, 'w'))

# Write the processed, grouped, and shuffled data to HDF5
for i in range(len(groups)):
    # Slice and shuffle PF candidates for Jet 1 and Jet 2
    j1pf_i = j1pf[groups[i]]  # select the events belonging to the current group
    j1pf_i = j1pf_i[orders[i]]  # sort them in the order of the current group
    out_files[i].create_dataset('jet1_PFCands', data = j1pf_i, chunks = (np.min([n_ev_gr[i], 1000]), 300), compression = 'gzip')  # save the data
    
    j2pf_i = j2pf[groups[i]]  
    j2pf_i = j2pf_i[orders[i]]  
    out_files[i].create_dataset('jet2_PFCands', data = j2pf_i, chunks = (np.min([n_ev_gr[i], 1000]), 300), compression = 'gzip')  # save the data
    
    # Slice and shuffle truth labels
    issignal_i = issignal[groups[i]]
    issignal_i = issignal_i[orders[i]]
    out_files[i].create_dataset('truth_label', data = issignal_i, compression = 'gzip')
    
    # Slice and shuffle jet kinematics
    jet_kinematics_i = jet_kinematics[groups[i]]
    jet_kinematics_i = jet_kinematics_i[orders[i]]
    out_files[i].create_dataset('jet_kinematics', data = jet_kinematics_i, chunks = (np.min([n_ev_gr[i], 1000]), 14), compression = 'gzip')

    # Slice and shuffle N-subjettiness features
    jettiness_i = jettiness_features[groups[i]]
    jettiness_i = jettiness_i[orders[i]]
    out_files[i].create_dataset('jettiness', data = jettiness_i, compression = 'gzip')

# Close all files and print metadata for verification
for out_file in out_files:
    out_keys = out_file.keys()
    print(out_keys)
    for key in out_keys:
        print(key)
        print(out_file[key])
    out_file.close()

