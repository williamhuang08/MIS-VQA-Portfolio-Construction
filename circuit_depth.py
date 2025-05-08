import pandas as pd
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

# Load correlation matrix and build full graph 
theta = 0.6
C = pd.read_csv("dow30_correlation_matrix.csv", index_col=0)
tickers = C.index.tolist()

G_full = nx.Graph()
G_full.add_nodes_from(tickers)
for i, ti in enumerate(tickers):
    for j in range(i+1, len(tickers)):
        tj = tickers[j]
        if abs(C.loc[ti, tj]) > theta:
            G_full.add_edge(ti, tj)

# Identify and remove isolated vertices
isolated = [node for node, deg in G_full.degree() if deg == 0]
print(f"Automatically in MIS (isolated): {isolated}")
G = G_full.copy()
G.remove_nodes_from(isolated)

reduced_tickers = list(G.nodes)
node2idx = {t: i for i, t in enumerate(reduced_tickers)}
n = len(reduced_tickers)

graph = {
    node2idx[t]: [node2idx[nb] for nb in G.neighbors(t)]
    for t in reduced_tickers
}

penalty = float(max(deg for _, deg in G.degree()))

# Build MIS Hamiltonian on reduced graph
def create_mis_hamiltonian(graph, penalty):
    obs, coeffs = [], []
    for i in range(len(graph)):
        obs.append(qml.PauliZ(i)); coeffs.append(0.5)
    for i, nbrs in graph.items():
        for j in nbrs:
            if j > i:
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(penalty / 4.0)
    return qml.Hamiltonian(coeffs, obs)

H = create_mis_hamiltonian(graph, penalty)

dev = qml.device("default.qubit", wires=n)
sampler_dev = qml.device("default.qubit", wires=n, shots=512)

# QAOA core circuits
def qaoa_expectation(params, p):
    gammas, betas = params[:p], params[p:2*p]
    # initial superposition
    for i in range(n):
        qml.Hadamard(wires=i)
    # p layers
    for layer in range(p):
        for coeff, op in zip(H.coeffs, H.ops):
            wires = op.wires
            angle = 2 * coeff * gammas[layer]
            if len(wires) == 1:
                qml.RZ(angle, wires=wires[0])
            else:
                qml.MultiRZ(angle, wires=wires)
        for i in range(n):
            qml.RX(2 * betas[layer], wires=i)
    return qml.expval(H)

def make_qnode(p):
    @qml.qnode(dev, interface="autograd")
    def cost_fn(params):
        return qaoa_expectation(params, p)
    return cost_fn

def make_sampler(p):
    @qml.qnode(sampler_dev, interface="autograd")
    def sampler(params):
        for i in range(n):
            qml.Hadamard(wires=i)
        gammas, betas = params[:p], params[p:2*p]
        for layer in range(p):
            for coeff, op in zip(H.coeffs, H.ops):
                wires = op.wires
                angle = 2 * coeff * gammas[layer]
                if len(wires) == 1:
                    qml.RZ(angle, wires=wires[0])
                else:
                    qml.MultiRZ(angle, wires=wires)
            for i in range(n):
                qml.RX(2 * betas[layer], wires=i)
        return [qml.sample(qml.PauliZ(i)) for i in range(n)]
    return sampler

# Iterative QAOA optimizer
def iterative_qaoa(graph, p, steps=200, lr=0.1):
    cost_fn = make_qnode(p)
    # p = 1 analytic (greedy equivalent)
    if p == 1:
        params = np.array([np.pi/4, np.pi/2], requires_grad=True)
        return params, cost_fn(params).item()
    prev_params, _ = iterative_qaoa(graph, p-1, steps, lr)
    init_guess = np.array([np.pi/4, np.pi/2])
    params = np.concatenate([prev_params, init_guess])
    params = np.array(params, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    for _ in range(steps):
        params = opt.step(cost_fn, params)
    return params, cost_fn(params).item()

for p in range(1, 6):
    print(f"\n=== Depth p = {p} ===")
    params, energy = iterative_qaoa(graph, p, steps=200, lr=0.2)
    print(f" Final ⟨H⟩ = {energy:.4f}")

    sampler = make_sampler(p)
    raw = np.array(sampler(params))          
    bitstr = ((1 - raw)//2).mean(axis=1).round().astype(int)

    # Map reduced bitstring back to full ticker list, include isolated
    mis_reduced = {reduced_tickers[i] for i, b in enumerate(bitstr) if b}
    final_mis = sorted(list(mis_reduced.union(isolated)))

    print(f" MIS size = {len(final_mis)} → {final_mis}")
