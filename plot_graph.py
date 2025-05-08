import yfinance as yf
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from corr_matrix import download_prices

corr = pd.read_csv("dow30_correlation_matrix.csv", index_col=0)

threshold = 0.6
G = nx.Graph()
tickers = corr.index.tolist()
G.add_nodes_from(tickers)

for i, ti in enumerate(tickers):
    for j in range(i+1, len(tickers)):
        tj = tickers[j]
        if abs(corr.loc[ti, tj]) > threshold:
            G.add_edge(ti, tj, weight=corr.loc[ti, tj])

pos = nx.spring_layout(G, seed=42, k=1)
nx.draw_networkx_nodes(G, pos, node_size=600)
pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > 0]
neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]
nx.draw_networkx_edges(G, pos, edgelist=pos_edges, style="solid")
nx.draw_networkx_edges(G, pos, edgelist=neg_edges, style="dashed")
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title(f"Market Graph ($\\theta$ = {threshold})")
plt.axis("off")
plt.tight_layout()

output_path = "market_graph.png"
plt.savefig(output_path, dpi=300)
print(f"Market graph saved to {output_path}")


