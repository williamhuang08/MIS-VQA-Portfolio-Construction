import pandas as pd
import networkx as nx
import pennylane as qml
from pennylane import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, PULP_CBC_CMD

theta = 0.6
C = pd.read_csv("dow30_correlation_matrix.csv", index_col=0)
nodes = C.index.tolist()

G_full = nx.Graph()
G_full.add_nodes_from(nodes)
for i, ti in enumerate(nodes):
    for tj in nodes[i+1:]:
        if abs(C.loc[ti, tj]) > theta:
            G_full.add_edge(ti, tj)

# 2) Solve exact MIS via ILP
prob = LpProblem("MaximumIndependentSet", LpMaximize)
x = LpVariable.dicts("x", nodes, cat="Binary")
prob += lpSum(x[i] for i in nodes)
for i, j in G_full.edges():
    prob += x[i] + x[j] <= 1
prob.solve(PULP_CBC_CMD(msg=False))

mis_exact = [i for i in nodes if x[i].value() == 1]
size_exact = len(mis_exact)
print(f"Exact MIS size = {size_exact}")

# 3) Remove isolated vertices and build reduced graph for QAOA
isolated = [n for n, d in G_full.degree() if d == 0]
G = G_full.copy()
G.remove_nodes_from(isolated)

reduced_nodes = list(G.nodes())
node2idx = {t: i for i, t in enumerate(reduced_nodes)}
graph = {
    node2idx[t]: [node2idx[nb] for nb in G.neighbors(t)]
    for t in reduced_nodes
}
n = len(reduced_nodes)
penalty = float(max(d for _, d in G.degree()))

# Build Hamiltonian
def create_mis_hamiltonian(graph, penalty):
    obs, coeffs = [], []
    for i in range(len(graph)):
        obs.append(qml.PauliZ(i)); coeffs.append(0.5)
    for i, nbrs in graph.items():
        for j in nbrs:
            if j > i:
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coeffs.append(penalty/4)
    return qml.Hamiltonian(coeffs, obs)

H = create_mis_hamiltonian(graph, penalty)

# QAOA core
dev = qml.device("default.qubit", wires=n)
sampler_dev = qml.device("default.qubit", wires=n, shots=512)

def qaoa_expectation(params, p):
    gammas, betas = params[:p], params[p:2*p]
    for i in range(n):
        qml.Hadamard(wires=i)
    for layer in range(p):
        for coeff, op in zip(H.coeffs, H.ops):
            angle = 2*coeff*gammas[layer]
            if len(op.wires)==1:
                qml.RZ(angle, wires=op.wires[0])
            else:
                qml.MultiRZ(angle, wires=op.wires)
        for i in range(n):
            qml.RX(2*betas[layer], wires=i)
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
                angle = 2*coeff*gammas[layer]
                if len(op.wires)==1:
                    qml.RZ(angle, wires=op.wires[0])
                else:
                    qml.MultiRZ(angle, wires=op.wires)
            for i in range(n):
                qml.RX(2*betas[layer], wires=i)
        return [qml.sample(qml.PauliZ(i)) for i in range(n)]
    return sampler

# Iterative QAOA
def iterative_qaoa(p, steps=200, lr=0.1):
    cost_fn = make_qnode(p)
    if p == 1:
        params = np.array([np.pi/4, np.pi/2], requires_grad=True)
    else:
        prev, _ = iterative_qaoa(p-1, steps, lr)
        params = np.concatenate([prev, np.array([np.pi/4, np.pi/2])])
        params = np.array(params, requires_grad=True)
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(steps):
            params = opt.step(cost_fn, params)
    energy = cost_fn(params).item()
    return params, energy

# Compare for different circuit depths
print("\n p |  QAOA size  |  overlap  |  ⟨H⟩ ")
print("---|-------------|-----------|--------")
for p in [1,2,3,4,5]:
    params, energy = iterative_qaoa(p, steps=200, lr=0.2)
    sampler = make_sampler(p)
    raw = np.array(sampler(params))
    bits = ((1-raw)//2).mean(axis=1).round().astype(int)
    mis_vqa = {reduced_nodes[i] for i,b in enumerate(bits) if b} | set(isolated)
    size_vqa = len(mis_vqa)
    inter = mis_vqa & set(mis_exact)
    union = mis_vqa | set(mis_exact)
    jaccard = len(inter)/len(union)
    print(f" {p} |     {size_vqa:2d}      |   {jaccard:.2f}   | {energy:.4f}")
