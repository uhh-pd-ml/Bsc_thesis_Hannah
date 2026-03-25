import matplotlib.pyplot as plt
import numpy as np

def plot_ae_simple(layers, n_inputs, save_name="ae_architektur"):
    all_layers = [n_inputs] + layers
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Koordinaten berechnen
    x_coords = range(len(all_layers))
    max_nodes = max(all_layers)
    
    for i, nodes in enumerate(all_layers):
        # Zentriere die Knoten vertikal
        y_coords = np.linspace(-nodes/2, nodes/2, nodes)
        
        # Zeichne Knoten
        color = 'orange' if nodes == min(layers) else 'skyblue'
        if i == 0: color = 'green'
        if i == len(all_layers)-1: color = 'red'
        
        ax.scatter([i]*nodes, y_coords, s=300, zorder=3, color=color, edgecolors='black')
        ax.text(i, max_nodes/2 + 0.5, f"Layer {i}\n({nodes})", ha='center', fontsize=10, fontweight='bold')

    ax.axis('off')
    plt.savefig(f"{save_name}.png", bbox_inches='tight')
    print(f"Grafik gespeichert unter: {save_name}.png")

# Aufruf
plot_ae_simple([12, 6, 2, 6, 12, 22], 22)