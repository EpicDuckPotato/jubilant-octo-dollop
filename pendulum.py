import numpy as np
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
from pydrake.symbolic import TaylorExpand, cos, sin

def symbolic_dynamics_u(state, t, u):
  th = state[0]
  thdot = state[1]
  c = cos(th)
  s = sin(th)
  thddot = u[0] - thdot - s
  return np.array([thdot, thddot])

def symbolic_dynamics(state, t, policy):
  th = state[0]
  thdot = state[1]
  u = policy.get_u(state, t)
  c = cos(th)
  s = sin(th)
  thddot = u[0] - thdot - s
  return np.array([thdot, thddot])

def dynamics(state, u):
  theta = state[0]
  thetadot = state[1]
  s = np.sin(theta)
  c = np.cos(theta)
  thetaddot = u[0] - thetadot - s
  return np.array([thetadot, thetaddot])

def linearize_dynamics(state, u):
  theta = state[0]
  thetadot = state[1]

  s = np.sin(theta)
  c = np.cos(theta)

  A = np.zeros((2, 2))
  B = np.zeros((2, 1))

  A[0, 1] = 1
  A[1, 0] = -c
  A[1, 1] = -1

  B[1, 0] = 1

  return A, B
