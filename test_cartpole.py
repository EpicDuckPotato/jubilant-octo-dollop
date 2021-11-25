from lqr_tree import LQRTree
import numpy as np
from cartpole import symbolic_dynamics, dynamics, linearize_dynamics
import pickle
import matplotlib.pyplot as plt

xmax = np.array([3, np.pi, 1, 1])
nx = 4
nu = 1
xgoal = np.array([0, np.pi, 0, 0])
ugoal = np.array([0])
ulb = -5
uub = 5
dt = 0.1
branch_horizon = 50
tree = LQRTree(xmax, symbolic_dynamics, dynamics, linearize_dynamics, nx, nu, xgoal, ugoal, ulb, uub, dt, branch_horizon)
tree.build_tree()
tree.plot_funnel(tree.nodes[1])
with open('tree.pkl', 'wb') as f:
  pickle.dump(tree, f)

xs, us = tree.trace(np.array([1, 3.4, 0, 0]))
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
