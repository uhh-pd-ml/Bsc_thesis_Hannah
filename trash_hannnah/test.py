import os
from pathlib import Path
import numpy as np
import torch
from hls4ml.utils.config import config_from_pytorch_model
from hls4ml.converters import convert_from_pytorch_model

class HLSModelExporter:
    def __init__(self, model, n_inputs=26, save_path="./"):
        self.model = model
        self.n_inputs = n_inputs
        # Wandelt den Pfad direkt in einen absoluten Systempfad um
        self.save_path = os.path.abspath(save_path)

    def export_to_hls(self, X_test_bkg, X_test_sig, output_dir_name="hls_project", backend='vitis', target='xcvu9p-flga2104-2L-e'):
        self.model.eval()
        self.model.to("cpu")
        
        # Generiere den finalen, absoluten Ordnerpfad
        output_dir = os.path.join(self.save_path, output_dir_name)
        
        # --- ABSOLUT SICHERES ERSTELLEN DES ORDNERS ---
        # Path().mkdir erstellt alle Unterordner (parents=True) und wirft keinen Fehler falls er existiert (exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Kontroll-Ausdruck im Terminal
        if os.path.exists(output_dir):
            print(f"[ERFOLG] Ordner wurde manuell angelegt unter: {output_dir}")
        else:
            raise OSError(f"[FEHLER] Ordner konnte nicht angelegt werden unter: {output_dir}")
        
        print("Starting hls4ml conversion...")
        
        # hls4ml Konfiguration erstellen
        hls_config = config_from_pytorch_model(
            self.model,
            input_shape=(None, self.n_inputs), 
            granularity='name',
            backend=backend,
            transpose_outputs=True
        )
        hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
        
        # WICHTIG: hls4ml intern mitteilen, wo es hinschreiben MUSS
        hls_config['OutputDir'] = output_dir 
        
        # Konvertierung starten
        hls_model = convert_from_pytorch_model(
            self.model,
            io_type='io_parallel',
            output_dir=output_dir, # Expliziter Pfad für die C++ Dateien
            backend=backend,
            hls_config=hls_config,
            part=target,
        )
        
        print("Kompiliere HLS Modell...")
        hls_model.compile()

        # Datencheck
        X_bkg_c = np.ascontiguousarray(X_test_bkg).astype(np.float32)
        p_hls_bkg = hls_model.predict(X_bkg_c)
        score_bkg_hls = np.mean((X_bkg_c - p_hls_bkg)**2, axis=-1)
        
        X_sig_c = np.ascontiguousarray(X_test_sig).astype(np.float32)
        p_hls_sig = hls_model.predict(X_sig_c)
        score_sig_hls = np.mean((X_sig_c - p_hls_sig)**2, axis=-1)
        
        return hls_model, score_bkg_hls, score_sig_hls