from polynomial_tree import PolynomialTree
import numpy as np
from cartpole import symbolic_dynamics_u, dynamics, linearize_dynamics
import pickle
import matplotlib.pyplot as plt

xmax = np.array([3, np.pi, 1, 1])
nx = 4
nu = 1
xgoal = np.array([0, np.pi, 0, 0])
ugoal = np.array([0])
ulb = -5*np.ones(1)
uub = 5*np.ones(1)
dt = 0.1
branch_horizon = 50
tree = PolynomialTree(xmax, symbolic_dynamics_u, dynamics, linearize_dynamics, nx, nu, xgoal, ugoal, ulb, uub, dt, branch_horizon)
tree.build_tree()
xlabel = 'x'
ylabel = 'theta'
tree.plot_all_funnels(xlabel, ylabel, 'cartpole_funnels.png')

xs, us = tree.trace(np.array([1, 3.4, 0, 0]), xlabel, ylabel, 'cartpole_trace.png')
plt.figure()
plt.plot([x[0] for x in xs])
plt.figure()
plt.plot([x[1] for x in xs])
plt.figure()
plt.plot([x[2] for x in xs])
plt.figure()
plt.plot([x[3] for x in xs])
plt.figure()
plt.plot(us)
plt.show()
