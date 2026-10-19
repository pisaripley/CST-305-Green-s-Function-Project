# CST-305 Project 3 - ODE settings
# Programmers: Pisa Ripley, Noah Calderon-Zuniga
# Packages: numpy
# Holds the two assigned ODEs (form y'' + p y' + q y = g(t)), their initial values
# and our hand solutions. GF1.py reads Part 1 from here, GF2.py reads the network part.
import numpy as np

# ---- Part 1 / Part 2: the two assigned ODEs ----
t0 = 0.0
t_end = 10.0

# ODE 1: y'' + y = 4,  t >= 0,  y(0) = y'(0) = 0
ode1 = {
    "text": "y'' + y = 4",
    "p": lambda t: 0.0,
    "q": lambda t: 1.0,
    "g": lambda t: 4.0,
    "y0": 0.0,
    "v0": 0.0,
    "hand": lambda t: 4 - 4*np.cos(t),          # our hand solution
}

# ODE 2: y'' + 4y = x,  x >= 0,  y(0) = y'(0) = 0   (x is the independent variable, written t here)
ode2 = {
    "text": "y'' + 4y = t",
    "p": lambda t: 0.0,
    "q": lambda t: 4.0,
    "g": lambda t: t,
    "y0": 0.0,
    "v0": 0.0,
    "hand": lambda t: t/4 - np.sin(2*t)/8,      # our hand solution
}

ODES = [ode1, ode2]


def read_odes():
    # reads the ODE as input: keep the assigned ones or type a new one
    print("Assigned ODEs:")
    for i, o in enumerate(ODES, 1):
        print(f"  {i}. {o['text']},  y({t0:g}) = {o['y0']:g},  y'({t0:g}) = {o['v0']:g}")
    if input("Type in a different ODE instead? (y/n): ").strip().lower() != "y":
        return ODES
    env = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp,
           "log": np.log, "sqrt": np.sqrt, "pi": np.pi}
    p_txt = input("p(t) = ")
    q_txt = input("q(t) = ")
    g_txt = input("g(t) = ")
    y0 = float(input(f"y({t0:g}) = "))
    v0 = float(input(f"y'({t0:g}) = "))
    return [{"text": f"y'' + ({p_txt})y' + ({q_txt})y = {g_txt}",
             "p": lambda t: eval(p_txt, env, {"t": t}),
             "q": lambda t: eval(q_txt, env, {"t": t}),
             "g": lambda t: eval(g_txt, env, {"t": t}),
             "y0": y0, "v0": v0, "hand": None}]


# ---- Network model (GF2.py, used for the lab question) ----
# u = data received at a node (MB), t in ms
# node is a dict with: hops, latency, bandwidth, data, loss, arrival
# p_net and q_net should only depend on the node, not on t
NET_ODE_TEXT = "u'' + 2w u' + w^2 u = w^2 * data * (1 - loss)"

def w(node):
    return 1.0 / (5 + 2*node["hops"])   # more hops = slower response

def p_net(node):
    return 2 * w(node)

def q_net(node):
    return w(node)**2

def g_net(t, node):
    return w(node)**2 * node["data"] * (1 - node["loss"]) * np.ones_like(t)

u0 = 0.0
du0 = 0.0
sim_time = 1500   # ms
data_size = 5.0   # MB