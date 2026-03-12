#!/usr/bin/env python
# coding: utf-8

import h5py    
import numpy as np 
#from pyjet import cluster,DTYPE_PTEPM
import fastjet
import pandas as pd
import awkward as ak
import vector


signal_region = (3150, 3850)


out_names = ['/beegfs/u/bbd1146/daten/events_b_br.h5', '/beegfs/u/bbd1146/daten/events_b_sr.h5', '/beegfs/u/bbd1146/daten/events_s_br.h5', '/beegfs/u/bbd1146/daten/events_s_sr.h5']


fnew = pd.read_hdf("/beegfs/u/bbd1146/daten/events_anomalydetection.h5")

events_combined = fnew 
events_np = np.asarray(events_combined)
n_pf = fnew.shape[0]  


# Create arrays for the PFCands for the first and the second jet
j1pf = np.zeros((n_pf, 300), np.float16)
j2pf = np.zeros((n_pf, 300), np.float16)

# an array for the jet kinematics variables
# 14 floats. Mjj, delta_eta (between j1 and j2), followed by the 4 vectors of j1, j2 and j3 in pt,eta,phi,m_softdrop format (if no 3rd jet passing cuts, zeros)
jet_kinematics = np.zeros((n_pf, 14), np.float64)


# In[9]:
vector.register_awkward()
jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)

events_np = np.asarray(events_combined)

print(fnew.shape)
print(events_combined.shape)
print(events_np.shape)


for i in range(len(events_np)):
    if i % 1000 == 0:
        print(f'> computed {i} events')

    event = events_np[i, :700*3].reshape(700, 3)
    mask  = event[:, 0] > 0
    pts   = event[mask, 0]
    etas  = event[mask, 1]
    phis  = event[mask, 2]

    array = ak.Array(
        {"pt": pts, "eta": etas, "phi": phis, "M": np.zeros(len(pts))},
        with_name="Momentum4D",
    )

    #Clustering
    cluster    = fastjet.ClusterSequence(array, jetdef)
    jets       = cluster.inclusive_jets(min_pt=0)
    sort_idx   = ak.argsort(jets.pt, ascending=False)
    jets, all_consts = jets[sort_idx], cluster.constituents()[sort_idx]
    

    n_jets      = len(jets)
    n_jets_save = min(n_jets, 3)

    #mjj
    if n_jets > 1:
        E  = jets[0].energy + jets[1].energy
        px = jets[0].px + jets[1].px
        py = jets[0].py + jets[1].py
        pz = jets[0].pz + jets[1].pz
        jet_kinematics[i, 0] = np.sqrt(max(E**2 - px**2 - py**2 - pz**2, 0.0))

    #Jet kinematics + PF candidates

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


# In[10]:


jet_kinematics[:, 1] = np.abs(jet_kinematics[:, 3] - jet_kinematics[:, 7])  # calculate Delta eta


# In[11]:


issignal = np.zeros((n_pf, 1), dtype = np.int8)
issignal[:, 0] = fnew.iloc[:, 2100]


# In[12]:


groups = []

groups.append(((jet_kinematics[:, 0] < signal_region[0]) | (jet_kinematics[:, 0] > signal_region[1])) & (issignal.reshape(-1) == 0))  # B in BR
groups.append((jet_kinematics[:, 0] >= signal_region[0]) & (jet_kinematics[:, 0] <= signal_region[1]) & (issignal.reshape(-1) == 0))  # B in SR

groups.append(((jet_kinematics[:, 0] < signal_region[0]) | (jet_kinematics[:, 0] > signal_region[1])) & issignal.reshape(-1))  # S in BR
groups.append((jet_kinematics[:, 0] >= signal_region[0]) & (jet_kinematics[:, 0] <= signal_region[1]) & issignal.reshape(-1))  # S in SR


# In[13]:


groups = np.array(groups, dtype = np.bool)
if ((groups.sum(0) != 1).sum()) != 0:
    raise AssertionError('Some elements are in no or more than one group')


# In[14]:


n_ev_gr = groups.sum(1)  # number of events in each group


# In[15]:


orders = []

for i in range(len(groups)):
    order = np.arange(n_ev_gr[i])
    np.random.shuffle(order)
    orders.append(order)
    with open('/beegfs/u/bbd1146/daten/shuffle_order_region_' + str(i) + '.npy', 'wb') as f_order:
        np.save(f_order, order)


# In[16]:


# open the files
out_files = []
for name in out_names:
    out_files.append(h5py.File(name, 'w'))


# In[17]:


for i in range(len(groups)):
    j1pf_i = j1pf[groups[i]]  # select the events belonging to the current group
    j1pf_i = j1pf_i[orders[i]]  # sort them in the order of the current group
    out_files[i].create_dataset('jet1_PFCands', data = j1pf_i, chunks = (np.min([n_ev_gr[i], 1000]), 300), compression = 'gzip')  # save the data
    
    j2pf_i = j2pf[groups[i]]  # select the events belonging to the current group
    j2pf_i = j2pf_i[orders[i]]  # sort them in the order of the current group
    out_files[i].create_dataset('jet2_PFCands', data = j2pf_i, chunks = (np.min([n_ev_gr[i], 1000]), 300), compression = 'gzip')  # save the data
    
    issignal_i = issignal[groups[i]]
    issignal_i = issignal_i[orders[i]]
    out_files[i].create_dataset('truth_label', data = issignal_i, compression = 'gzip')
    
    jet_kinematics_i = jet_kinematics[groups[i]]
    jet_kinematics_i = jet_kinematics_i[orders[i]]
    out_files[i].create_dataset('jet_kinematics', data = jet_kinematics_i, chunks = (np.min([n_ev_gr[i], 1000]), 14), compression = 'gzip')


# In[18]:


for out_file in out_files:
    out_keys = out_file.keys()
    print(out_keys)
    for key in out_keys:
        print(key)
        print(out_file[key])
    out_file.close()

