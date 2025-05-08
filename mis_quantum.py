import pandas as pd
import networkx as nx
import pennylane as qml
from pennylane import numpy as np

theta = 0.6  
C = pd.read_csv("dow30_correlation_matrix.csv", index_col=0)  
tickers = C.index.tolist()

# Build NX graph
G_nx = nx.Graph()
G_nx.add_nodes_from(tickers)
for i, ti in enumerate(tickers):
    for j in range(i+1, len(tickers)):
        tj = tickers[j]
        if abs(C.loc[ti, tj]) > theta:
            G_nx.add_edge(ti, tj)

node2idx = {t: i for i, t in enumerate(tickers)}
n = len(tickers)
graph = { node2idx[t]: [node2idx[nb] for nb in G_nx.neighbors(t)] 
          for t in tickers }

max_deg = max(dict(G_nx.degree()).values())
penalty = float(max_deg)

# Build MIS Hamiltonian
def create_mis_hamiltonian(graph, penalty):
    obs, coeffs = [], []
    for i in range(len(graph)):
        obs.append(qml.PauliZ(i))
        coeffs.append(0.5)
    for i, nbrs in graph.items():
        for j in nbrs:
            if j > i:
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(penalty / 4.0)
    return qml.Hamiltonian(coeffs, obs)

H = create_mis_hamiltonian(graph, penalty=penalty)

# Devices
dev = qml.device("default.qubit", wires=n)            
sampler_dev = qml.device("default.qubit", wires=n, shots=512)

# QAOA expectation circuit 
def qaoa_expectation(params, p):
    gammas = params[:p]
    betas  = params[p:2*p]
    for i in range(n):
        qml.Hadamard(wires=i)
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

# Iterative QAOA with analytic p=1 
def iterative_qaoa(graph, max_p=3, steps_per_layer=200, lr=0.1):
    params = None
    for p in range(1, max_p+1):
        cost_fn = make_qnode(p)

        if p == 1:
            params = np.array([np.pi/4, np.pi/2], requires_grad=True)
            energy = cost_fn(params).item()
            print(f"Depth p=1 (analytic) → ⟨H⟩ = {energy:.4f}")
            continue

        init_guess = np.array([np.pi/4, np.pi/2])
        params = np.concatenate([params, init_guess])
        params = np.array(params, requires_grad=True)

        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(steps_per_layer):
            params = opt.step(cost_fn, params)

        energy = cost_fn(params).item()
        print(f"Depth p={p} (optimized) → ⟨H⟩ = {energy:.4f}")

    return params

# Sampler QNode
def make_sampler(p):
    @qml.qnode(sampler_dev, interface="autograd")
    def sampler(params):
        for i in range(n):
            qml.Hadamard(wires=i)
        gammas = params[:p]
        betas  = params[p:2*p]
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

if __name__ == "__main__":
    p_max = 3
    lr = 0.2
    steps = 200

    params = iterative_qaoa(graph, max_p=p_max, steps_per_layer=steps, lr=lr)

    sampler = make_sampler(p_max)
    raw = np.array(sampler(params))  

    # majority vote → bitstring
    bitstr = ((1 - raw)//2).mean(axis=1).round().astype(int)

    mis = [tickers[i] for i, b in enumerate(bitstr) if b]
    print(f"Approximate MIS size = {len(mis)}")
    print("Selected tickers:", mis)
