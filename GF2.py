# CST-305 Project 3 - Network data propagation (supports the lab question)
# Programmers: Pisa Ripley, Noah Calderon-Zuniga
# Packages: numpy, scipy, networkx, matplotlib
# Approach: finds the fastest path from the server to each node (networkx, Dijkstra), works out
# hops, latency, bandwidth, data size and packet loss, then solves each node's ODE from
# ODEConfig.py, u'' + p u' + q u = g(t - arrival), with Green's function (G = y2(t-s), a
# convolution). Prints a table and animates the data spreading like isotherms.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ODEConfig as ode

# network: (node, node, latency ms, bandwidth Mbps)
links = [
    ("Server", "Router1", 1, 1000),
    ("Server", "Router2", 1, 1000),
    ("Router1", "PC1", 2, 100),
    ("Router1", "PC2", 2, 100),
    ("Router2", "PC3", 3, 100),
    ("Router2", "WiFi", 2, 300),
    ("WiFi", "Laptop", 5, 54),
]
loss_rate = {"Server": 0, "Router1": 0.001, "Router2": 0.001, "PC1": 0.005,
             "PC2": 0.005, "PC3": 0.01, "WiFi": 0.02, "Laptop": 0.03}

G = nx.Graph()
for a, b, lat, bw in links:
    send_time = ode.data_size * 8 / bw * 1000   # ms to push the data through this link
    G.add_edge(a, b, latency=lat, bandwidth=bw, delay=lat + send_time)

# find the fastest path from the server to each node and add up the factors
paths = nx.shortest_path(G, "Server", weight="delay")
nodes = {}
for name, path in paths.items():
    latency, delay, bw, kept = 0, 0, 10**9, 1 - loss_rate["Server"]
    for a, b in zip(path, path[1:]):
        latency += G[a][b]["latency"]
        delay += G[a][b]["delay"]
        bw = min(bw, G[a][b]["bandwidth"])
        kept *= 1 - loss_rate[b]
    nodes[name] = {"hops": len(path) - 1, "latency": latency, "bandwidth": bw,
                   "data": ode.data_size, "loss": 1 - kept, "arrival": delay}

t = np.linspace(0, ode.sim_time, 3001)
dt = t[1] - t[0]


def solve_node(node):
    p, q = ode.p_net(node), ode.q_net(node)

    def homogeneous(t, Y):
        return [Y[1], -p*Y[1] - q*Y[0]]

    y1 = solve_ivp(homogeneous, (0, t[-1]), [1, 0], t_eval=t, rtol=1e-9).y[0]
    y2 = solve_ivp(homogeneous, (0, t[-1]), [0, 1], t_eval=t, rtol=1e-9).y[0]

    # input only starts once the data reaches the node
    gvals = np.where(t >= node["arrival"], ode.g_net(t - node["arrival"], node), 0)

    # p and q are constant so G(t,s) = y2(t-s), and the Green's integral is a convolution
    forced = np.convolve(gvals, y2)[:len(t)] * dt
    return ode.u0*y1 + ode.du0*y2 + forced


results = {name: solve_node(node) for name, node in nodes.items()}

print("ODE:", ode.NET_ODE_TEXT)
print(f"data size = {ode.data_size} MB\n")
print("node      hops  latency(ms)  bandwidth(Mbps)  arrival(ms)  received(MB)")
for name in sorted(nodes, key=lambda n: nodes[n]["arrival"]):
    n = nodes[name]
    bw = "-" if name == "Server" else n["bandwidth"]
    print(f"{name:<9} {n['hops']:>4} {n['latency']:>11} {bw:>16} {n['arrival']:>12.1f} {results[name][-1]:>13.3f}")

# simple animation: node color shows how much data it has
pos = nx.spring_layout(G, seed=1)
order = list(G.nodes)
for i in range(0, len(t), 100):
    plt.clf()
    colors = [results[n][i] for n in order]
    nx.draw(G, pos, nodelist=order, node_color=colors, cmap="Blues", vmin=0,
            vmax=ode.data_size, with_labels=True, node_size=1200, edgecolors="black")
    plt.title(f"Data propagation, t = {t[i]:.0f} ms")
    plt.pause(0.05)
plt.show()

# data received over time
plt.figure()
for name in results:
    plt.plot(t, results[name], label=name)
plt.xlabel("time (ms)")
plt.ylabel("data received (MB)")
plt.title("Data received at each node")
plt.legend()
plt.grid()
plt.show()