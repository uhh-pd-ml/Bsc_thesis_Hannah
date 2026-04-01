import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('/beegfs/u/bbd1146/daten_test/events_s_sr.h5', 'r') as f:
    print("Keys in File:", f.keys())
    data = f['jettiness'][:]
    print("Shape:", data.shape)
    print("Beispielwerte (erstes Event):\n", data[0])

    kin_data = f['jet_kinematics'][:]
    print("\n--- Kinematik Test ---")
    print("Shape:", kin_data.shape)
    
    # Oft sind die Daten (Events, Partikel, Features)
    # Beispiel: (10000, 20, 4) -> 10k Events, 20 Partikel, 4 Features (px, py, pz, E)
    print("Erstes Event (alle Partikel/Features):\n", kin_data[0])
    
    # Statistische Kurzprüfung
    print("Mittelwert pT/Feature 0:", np.mean(kin_data[:]))

with h5py.File('/beegfs/u/bbd1146/daten_test/events_s_sr.h5', 'r') as f:
    kin = f['jet_kinematics'][:]
    tau = f['jettiness'][:]

    # Check 1: Sind Massen positiv?
    print(f"Negative Massen? {np.any(kin[:, [5, 9]] < 0)}")

    # Check 2: Verteilung der Jet-Masse (Spalte 5)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(kin[:, 5], bins=50, color='blue', alpha=0.7)
    plt.title("Jet 1 Masse (Feature 5)")
    plt.xlabel("Masse [GeV]")

    # Check 3: Verteilung Jettiness (Spalte 0)
    plt.subplot(1, 2, 2)
    plt.hist(tau[:, 0], bins=50, color='green', alpha=0.7)
    plt.title("Tau_1 (Jettiness)")
    plt.xlabel("Wert")
    plt.savefig('jet_check.png')



'''
# in for Schleife machen:
if i == 100: 
        print(f"Testlauf: Breche bei Event {i} ab.")
        # Kürze alle bisher definierten Arrays auf die Test-Größe
        j1pf = j1pf[:i]
        j2pf = j2pf[:i]
        jet_kinematics = jet_kinematics[:i]
        jettiness_features = jettiness_features[:i]
        issignal = issignal[:i]
        n_pf = i  # Wichtig, damit die Gruppenberechnung später stimmt
        break
'''