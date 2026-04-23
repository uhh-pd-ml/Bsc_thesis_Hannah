import pandas as pd
import numpy as np
import h5py

def create_option_c_dataset(input_path, output_path):
    df = pd.read_hdf(input_path)
    events_np = np.asarray(df)
    n_events = events_np.shape[0]
    
    
    processed_data = np.zeros((n_events, 60))
    
    for i in range(n_events):
        
        event = events_np[i, :700*3].reshape(700, 3)
        
        idx = np.argsort(event[:, 0])[::-1]
        top_20 = event[idx][:20, :] 
        processed_data[i] = top_20.flatten()
        
    
    np.save(output_path, processed_data.astype(np.float32))


create_option_c_dataset('/beegfs/u/bbd1146/events_anomalydetection.h5', 'X_train_optC.npy')