import numpy as np
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
from pydrake.symbolic import TaylorExpand, cos, sin

def symbolic_dynamics_u(state, t, u):
  x = state[0]
  th = state[1]
  xdot = state[2]
  thdot = state[3]
  c = cos(th)
  s = sin(th)
  xddot = (u[0] + s*c + thdot**2*s)/(2 - c**2)
  thddot = -xddot*c - s
  return np.array([xdot, thdot, xddot, thddot])

def symbolic_dynamics(state, t, policy):
  x = state[0]
  th = state[1]
  xdot = state[2]
  thdot = state[3]
  u = policy.get_u(state, t)
  c = cos(th)
  s = sin(th)
  xddot = (u[0] + s*c + thdot**2*s)/(2 - c**2)
  thddot = -xddot*c - s
  return np.array([xdot, thdot, xddot, thddot])

def dynamics(state, u):
  x = state[0]
  theta = state[1]
  xdot = state[2]
  thetadot = state[3]
  s = np.sin(theta)
  c = np.cos(theta)
  xddot = (u[0] + s*c + thetadot**2*s)/(2 - c**2)
  thetaddot = -xddot*c - s
  return np.array([xdot, thetadot, xddot, thetaddot])

def linearize_dynamics(state, u):
  x = state[0]
  theta = state[1]
  xdot = state[2]
  thetadot = state[3]

  s = np.sin(theta)
  c = np.cos(theta)

  xddot = (u + s*c + thetadot**2*s)/(2 - c**2)

  A = np.zeros((4, 4))
  B = np.zeros((4, 1))

  A[:2, 2:] = np.eye(2)
  A[2, 1] = ((2 - c**2)*(-s**2 + c**2 + thetadot**2*c) - (u + s*c + thetadot**2*s)*2*s*c)/(2 - c**2)**2
  A[2, 3] = 2*thetadot*s/(2 - c**2)
  A[3, 1] = -A[2, 1]*c + xddot*s - c
  A[3, 3] = -c*A[2, 3]

  B[2] = 1/(2 - c**2)
  B[3] = -B[2]*c

  return A, B
