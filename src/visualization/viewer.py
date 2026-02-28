import torch
import plotly.graph_objects as go
from leviathan.config import REGIONS

def visualize_topology(positions: torch.Tensor, region_labels: list, weights: torch.Tensor, show_edges=True):
    """
    Visualizes the generated 3D physical network using Plotly.
    Colors by Brain Region.
    """
    fig = go.Figure()

    # Assign colors to regions
    colors = {
        'thalamus': 'red',
        'cortex_v': 'blue',
        'cortex_p': 'cyan',
        'cerebellum': 'purple',
        'limbic': 'green'
    }

    # Edges (Sparse CSR plotting)
    if show_edges:
        edge_x = []
        edge_y = []
        edge_z = []
        
        N = weights.shape[0]
        # For visualization, only show strong synapses
        # Or a random subsample if N is large
        
        # Get indices of active synapses
        sources, targets = torch.nonzero(weights, as_tuple=True)
        
        # Max edges to draw to prevent lagging browser
        max_edges = min(len(sources), 5000)
        
        for idx in range(max_edges):
            s = sources[idx]
            t = targets[idx]
            
            p_s = positions[s]
            p_t = positions[t]
            
            edge_x.extend([p_s[0].item(), p_t[0].item(), None])
            edge_y.extend([p_s[1].item(), p_t[1].item(), None])
            edge_z.extend([p_s[2].item(), p_t[2].item(), None])
            
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(150,150,150,0.2)', width=1),
            hoverinfo='none',
            name='Axons'
        ))

    # Nodes
    unique_regions = list(set(region_labels))
    for region in unique_regions:
        indices = [i for i, x in enumerate(region_labels) if x == region]
        pos = positions[indices]
        
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0].numpy(),
            y=pos[:, 1].numpy(),
            z=pos[:, 2].numpy(),
            mode='markers',
            marker=dict(
                size=4,
                color=colors.get(region, 'black'),
                opacity=0.8
            ),
            name=region.capitalize()
        ))

    fig.update_layout(
        title="Leviathan v2.0 Connectome Topology",
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save to disk
    fig.write_html("leviathan_topology.html")
    print("Saved visualization to leviathan_topology.html")

if __name__ == '__main__':
    from leviathan.topology import generate_spatial_topology, initialize_adjacency_matrix
    from leviathan.config import TOPOLOGY_LAMBDA
    
    print("Generating 500 node topology...")
    pos, omegas, labels = generate_spatial_topology(500)
    adj = initialize_adjacency_matrix(pos, TOPOLOGY_LAMBDA)
    
    print("Visualizing...")
    visualize_topology(pos, labels, adj)
