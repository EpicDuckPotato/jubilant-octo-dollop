import sympy
import itertools
import mosek
from mosek.fusion import *
import numpy as np
import pickle
import re
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
from pydrake.symbolic import TaylorExpand, cos, sin, Monomial, Expression
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

def Vconstraint(V, xerrdot, xerr, la, rho):
  Vdot = V.Jacobian(xerr) @ xerrdot
  return -Vdot - la[0]*(rho - V)

# LQR stuff
A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 2, 0, 0]])
B = np.array([0, 0, 1, 1]).reshape(-1, 1)
R = np.eye(1)
Q = np.eye(4)
S = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, np.matmul(B.transpose(), S))
xd = np.array([0, np.pi, 0, 0])

def dynamics_u(state, u):
  x = state[0]
  th = state[1]
  xdot = state[2]
  thdot = state[3]
  c = cos(th)
  s = sin(th)
  xddot = (u[0] + s*c + thdot**2*s)/(2 - c**2)
  thddot = -xddot*c - s
  return np.array([xdot, thdot, xddot, thddot])

prog = MathematicalProgram()
xerr = prog.NewIndeterminates(4, 'xerr') 
usol = -K@xerr
x = xerr + xd
xdotsol = dynamics_u(x, usol)
for i in range(xdotsol.shape[0]):
  xdotsol[i] = TaylorExpand(xdotsol[i], {var: 0 for var in xerr}, 3)

Vsol = xerr@S@xerr

la1poly = prog.NewSosPolynomial(Variables(xerr), 2)[0] # ROA constraint
la1 = la1poly.ToExpression()

bestrho = 0
rho = 1
prevrho = rho
max_improve = 10
i = 0
# Line search for rho
while rho > 0.001:
  if i > max_improve:
    break

  print('Starting iteration ' + str(i) + ' with rho = ' + str(rho))

  prog_rho = prog.Clone()
  prog_rho.AddSosConstraint(Vconstraint(Vsol, xdotsol, xerr, [la1], rho))
  result = Solve(prog_rho)

  if result.is_success():
    print('Feasible with rho = ' + str(rho))
    bestrho = rho
    rho = (rho + prevrho)/2
    la1sol = result.GetSolution(la1poly).ToExpression()
  else:
    prevrho = rho
    rho = (rho + bestrho)/2

  i += 1

bestrho_pre_alt = bestrho

VQsol = S
Vpoly = prog.NewSosPolynomial([Monomial(xerri, 1) for xerri in xerr])
V = Vpoly[0].ToExpression()

upoly = [prog.NewFreePolynomial(Variables(xerr), 3)]
u = [poly.ToExpression() for poly in upoly]

la2poly = prog.NewSosPolynomial(Variables(xerr), 2)[0] # Actuator upper bound
la2 = la2poly.ToExpression()
la3poly = prog.NewSosPolynomial(Variables(xerr), 2)[0] # Actuator lower bound
la3 = la2poly.ToExpression()

desV1 = np.ones(4)@S@np.ones(4)

max_alternations = 10
max_inner = 10
for a in range(max_alternations):
  print('Optimizing controller')
  rhostep = bestrho
  rho = 2*bestrho
  xdot = dynamics_u(x, [poly.ToExpression() for poly in upoly])
  for i in range(xdot.shape[0]):
    xdot[i] = TaylorExpand(xdot[i], {var: 0 for var in xerr}, 3)

  for inner in range(max_inner):
    prog_rho = prog.Clone()
    prog_rho.AddSosConstraint(Vconstraint(Vsol, xdot, xerr, [la1], rho))

    # Actuator limits
    prog_rho.AddSosConstraint(10 - u[0] - la2*(rho - Vsol))
    prog_rho.AddSosConstraint(u[0] + 10 - la3*(rho - Vsol))

    result = Solve(prog_rho)

    if result.is_success():
      bestrho = rho
      rho = bestrho + rhostep
      usol = [result.GetSolution(poly).ToExpression() for poly in upoly]
      xdotsol = dynamics_u(x, usol)
      la2sol = result.GetSolution(la2poly).ToExpression()
      la3sol = result.GetSolution(la3poly).ToExpression()
    else:
      rhostep /= 2
      rho = bestrho + rhostep

  for i in range(xdotsol.shape[0]):
    xdotsol[i] = TaylorExpand(xdotsol[i], {var: 0 for var in xerr}, 3)

  print('New optimal rho of ' + str(bestrho))

  rhostep = bestrho
  rho = 2*bestrho

  print('Optimizing Lyapunov function')
  for inner in range(max_inner):
    prog_rho = prog.Clone()
    # Usual Lyapunov constraint
    prog_rho.AddSosConstraint(Vconstraint(V, xdotsol, xerr, [la1sol], rho))
    # Normalization constraint
    prog_rho.AddLinearConstraint(V.Substitute({xerr[i]: 1 for i in range(4)}) == desV1)
    result = Solve(prog_rho)

    # Actuator limits
    prog_rho.AddSosConstraint(10 - usol[0] - la2sol*(rho - V))
    prog_rho.AddSosConstraint(usol[0] + 10 - la3sol*(rho - V))

    if result.is_success():
      bestrho = rho
      rho = bestrho + rhostep
      Vsol = result.GetSolution(Vpoly[0]).ToExpression()
      VQsol = result.GetSolution(Vpoly[1]).ToExpression()
      la2sol = result.GetSolution(la2poly).ToExpression()
      la3sol = result.GetSolution(la3poly).ToExpression()
    else:
      rhostep /= 2
      rho = bestrho + rhostep

  print('New optimal rho of ' + str(bestrho))

# Plot pre-alternations version
# Project S to 2D using Schur complement
J = S[:2, :2]
L = S[2:, :2]
K = S[2:, 2:]
Sp = J - L.transpose()@np.linalg.solve(K, L)
theta = np.linspace(0, 2*np.pi, 100)
points = np.sqrt(bestrho_pre_alt)*np.stack((np.cos(theta), np.sin(theta)), 0)
L = np.linalg.cholesky(Sp)
points = np.linalg.solve(L.transpose(), points)

points[1] += np.pi
plt.plot(points[0], points[1], color='g', label='ROA pre-alternations')

# Plot post-alternations version
J = VQsol[:2, :2]
L = VQsol[2:, :2]
K = VQsol[2:, 2:]
Sp = J - L.transpose()@np.linalg.solve(K, L)
theta = np.linspace(0, 2*np.pi, 100)
points = np.sqrt(bestrho)*np.stack((np.cos(theta), np.sin(theta)), 0)
L = np.linalg.cholesky(Sp)
points = np.linalg.solve(L.transpose(), points)

points[1] += np.pi
plt.plot(points[0], points[1], color='g', label='ROA post-alternations')


plt.xlabel('x')
plt.ylabel('theta')
plt.title('Alternations Performance')
plt.legend()
plt.savefig('cartpole_alternations.png')
plt.show()
