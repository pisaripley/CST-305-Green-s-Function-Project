# CST-305 Project 3 - Green's Function and ODE with IVP
# Programmers: Pisa Ripley, Noah Calderon-Zuniga
# Packages: numpy, scipy, matplotlib
# Approach: reads the two assigned ODEs from ODEConfig.py (or one typed in when it runs).
# For each ODE y'' + p y' + q y = g(t):
#   1) solve the homogeneous ODE with scipy's solve_ivp to get y1 and y2, and plot them
#   2) Wronskian W = y1 y2' - y1' y2, Green's function G(t,s) = (y1(s)y2(t) - y1(t)y2(s)) / W(s)
#   3) y(t) = y0 y1 + v0 y2 + integral of G(t,s) g(s) ds   (Green's function solution)
#   4) compare with our hand solution and with scipy's RK45, then plot and animate it

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
import ODEConfig as cfg

odes = cfg.read_odes()
t = np.linspace(cfg.t0, cfg.t_end, 2001)
results = []

for k, ode in enumerate(odes, 1):
    p, q, g = ode["p"], ode["q"], ode["g"]

    def homogeneous(t, Y):
        return [Y[1], -p(t)*Y[1] - q(t)*Y[0]]

    # homogeneous part: y1(0)=1, y1'(0)=0 and y2(0)=0, y2'(0)=1
    s1 = solve_ivp(homogeneous, (t[0], t[-1]), [1, 0], t_eval=t, rtol=1e-10, atol=1e-12)
    s2 = solve_ivp(homogeneous, (t[0], t[-1]), [0, 1], t_eval=t, rtol=1e-10, atol=1e-12)
    y1, dy1 = s1.y
    y2, dy2 = s2.y

    # Wronskian
    W = y1*dy2 - dy1*y2

    # Green's function integral: G(t,s)g(s) splits into two integrals, done in one pass
    gvals = np.asarray(g(t), dtype=float) * np.ones_like(t)
    I1 = cumulative_trapezoid(y1*gvals/W, t, initial=0)
    I2 = cumulative_trapezoid(y2*gvals/W, t, initial=0)
    y = ode["y0"]*y1 + ode["v0"]*y2 + y2*I1 - y1*I2

    # check with RK45 on the full equation
    def full_ode(t, Y):
        return [Y[1], g(t) - p(t)*Y[1] - q(t)*Y[0]]

    rk = solve_ivp(full_ode, (t[0], t[-1]), [ode["y0"], ode["v0"]], t_eval=t,
                   rtol=1e-10, atol=1e-12).y[0]
    hand = ode["hand"](t) if ode["hand"] else None

    print(f"\nODE {k}: {ode['text']},  y({cfg.t0:g}) = {ode['y0']:g},  y'({cfg.t0:g}) = {ode['v0']:g}")
    print(f"Wronskian W({cfg.t0:g}) = {W[0]:.4f}")
    print(f"homogeneous solution with these initial values: y_h = {ode['y0']:g}*y1 + {ode['v0']:g}*y2")
    print("     t        y1        y2     Green's      hand      RK45")
    for i in range(0, len(t), 200):
        h = f"{hand[i]:9.5f}" if hand is not None else "      -  "
        print(f"{t[i]:6.2f} {y1[i]:9.5f} {y2[i]:9.5f} {y[i]:10.5f} {h} {rk[i]:9.5f}")
    print("max difference vs RK45:", np.max(np.abs(y - rk)))
    if hand is not None:
        print("max difference vs hand solution:", np.max(np.abs(y - hand)))

    results.append({"text": ode["text"], "y1": y1, "y2": y2, "y": y, "rk": rk, "hand": hand})

# plot 1: homogeneous solutions
fig, axes = plt.subplots(len(results), 1, figsize=(8, 3.5*len(results)), squeeze=False)
for ax, r in zip(axes[:, 0], results):
    ax.plot(t, r["y1"], label="y1")
    ax.plot(t, r["y2"], label="y2")
    ax.set_title("Homogeneous solutions of " + r["text"])
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()

# plot 2: Green's function solution vs hand solution and RK45
fig, axes = plt.subplots(len(results), 1, figsize=(8, 3.5*len(results)), squeeze=False)
for ax, r in zip(axes[:, 0], results):
    ax.plot(t, r["y"], label="Green's function")
    if r["hand"] is not None:
        ax.plot(t, r["hand"], "--", label="hand solution")
    ax.plot(t, r["rk"], ":", label="RK45")
    ax.set_title("Green's function solution of " + r["text"])
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.legend()
    ax.grid()
plt.tight_layout()
plt.show()

# simple animation: the Green's function solutions being drawn over time
fig, axes = plt.subplots(len(results), 1, figsize=(8, 3.5*len(results)), squeeze=False)
lines = []
for ax, r in zip(axes[:, 0], results):
    ax.set_xlim(t[0], t[-1])
    pad = 0.1*(r["y"].max() - r["y"].min() + 1e-9)
    ax.set_ylim(r["y"].min() - pad, r["y"].max() + pad)
    ax.set_title(r["text"])
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.grid()
    lines.append(ax.plot([], [], lw=2)[0])
plt.tight_layout()
for i in range(0, len(t) + 1, 40):
    for line, r in zip(lines, results):
        line.set_data(t[:i], r["y"][:i])
    plt.pause(0.03)
plt.show()