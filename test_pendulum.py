from lqr_tree import LQRTree
import numpy as np
from pendulum import symbolic_dynamics, dynamics, linearize_dynamics
import pickle
import matplotlib.pyplot as plt

xmax = np.array([np.pi, 10])
nx = 2
nu = 1
xgoal = np.array([np.pi, 0])
ugoal = np.array([0])
ulb = -5*np.ones(1)
uub = 5*np.ones(1)
dt = 0.1
branch_horizon = 50
tree = LQRTree(xmax, symbolic_dynamics, dynamics, linearize_dynamics, nx, nu, xgoal, ugoal, ulb, uub, dt, branch_horizon)
tree.build_tree()
xlabel = 'theta'
ylabel = 'theta_dot'
tree.plot_all_funnels(xlabel, ylabel, 'pendulum_funnels.png')

#xs, us = tree.trace(np.array([3.4, 12.6]))
xs, us = tree.trace(np.array([3.4, 7]), xlabel, ylabel, 'pendulum_trace.png')
plt.figure()
plt.plot([x[0] for x in xs])
plt.figure()
plt.plot([x[1] for x in xs])
plt.figure()
plt.plot(us)
plt.show()
