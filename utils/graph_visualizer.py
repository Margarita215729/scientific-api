import networkx as nx
import matplotlib.pyplot as plt
import base64
import io

def generate_graph_image(data):
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return {
        "image_base64": image_b64,
        "python_code": f"# nodes = {nodes}\n# edges = {edges}\n# Rendered with networkx"
    }
