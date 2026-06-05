import pandas as pd
import numpy as np
import h5py


df = pd.read_hdf('/beegfs/u/bbd1146/events_anomalydetection.h5')
events_np = np.asarray(df)
n_events = events_np.shape[0]


processed_data = np.zeros((n_events, 60))

for i in range(n_events):
    
    event = events_np[i, :700*3].reshape(700, 3)
    
    idx = np.argsort(event[:, 0])[::-1]
    top_20 = event[idx][:20, :] 
    processed_data[i] = top_20.flatten()
    

np.save('X_train_optC.npy', processed_data.astype(np.float32))