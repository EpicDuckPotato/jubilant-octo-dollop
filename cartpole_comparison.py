from cartpole import *
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

dt = 0.01

# integrate: integrates x forward for dt seconds
# ARGUMENTS
# x: x[0] is theta, x[1] is theta_dot
# u: actuator input
# RETURN
# x integrated forward for dt seconds
def integrate(x, u):
  x1 = x
  k1 = dynamics(x1, u)
  x2 = x + dt*k1/2
  k2 = dynamics(x2, u)
  x3 = x + dt*k2/2
  k3 = dynamics(x3, u)
  x4 = x + dt*k3
  k4 = dynamics(x4, u)
  return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

nx = 4
xd = np.array([0, np.pi, 0, 0])
ud = np.zeros(1)
A, B = linearize_dynamics(xd, ud)
Q = np.eye(nx)
R = np.eye(1)

S = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.transpose()@S)

prog = MathematicalProgram()
xerr = prog.NewIndeterminates(nx, 'xerr')
x = xerr + xd
u = -K@xerr
xerrdot = symbolic_dynamics_u(x, 0, u)
for i in range(nx):
  xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, 3) 
V = xerr@S@xerr
Vdot = 2*xerr@S@xerrdot

la = prog.NewSosPolynomial(Variables(xerr), 2)[0].ToExpression()

# Line search
lower = 0
rho = 1
upper = rho*2
max_improve = 8
rho_min = 1e-3
i = 0
while rho > rho_min:
  print('ROA line search iteration %d, testing rho = %f' %(i, rho))
  if i > max_improve and lower != 0:
    break

  prog_clone = prog.Clone()
  prog_clone.AddSosConstraint(-Vdot - la*(rho - V) - 0.001*xerr@xerr)

  result = Solve(prog_clone)

  if result.is_success():
    lower = rho
    rho = (rho + upper)/2
  else:
    upper = rho
    rho = (rho + lower)/2

  i += 1

if lower == 0:
  print('No region of attraction')

rho = lower
print('Finished ROA line search with rho = %f' %(rho))

# Now find actual ROA (do analysis in (x, theta) plane, letting xdot and thetadot = 0)
x = np.linspace(-7, 7, 100)
theta = np.linspace(0, 2*np.pi, 100)

x = np.stack(np.meshgrid(x, theta), 0)
x = np.concatenate((x, np.zeros_like(x)))
xd = np.tile(xd.reshape(4, 1, 1), (1, 100, 100))
u = np.zeros((1, 100, 100))
for step in range(5000):
  if step%100 == 0:
    print('Simulation step %d' %(step))
  errs = x - xd
  for row in range(100):
    for col in range(100):
      u[0, row, col] = -K@errs[:, row, col]
  x = integrate(x, u)

errs = x - xd
stable_idx = np.abs(errs[0]) < 0.01
for i in range(1, nx):
  stable_idx = np.logical_and(stable_idx, np.abs(errs[1]) < 0.01)
image = np.zeros((100, 100, 3))
image[stable_idx] = 1
plt.imshow(image, extent=[-7, 7, 0, 2*np.pi], aspect='auto', origin='lower')

J = S[:2, :2]
L = S[2:, :2]
K = S[2:, 2:]
Sp = J - L.transpose()@np.linalg.solve(K, L)

theta = np.linspace(0, 2*np.pi, 100)
points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
L = np.linalg.cholesky(Sp)
points = np.linalg.solve(L.transpose(), points)

for i in range(2):
  points[i] += xd[i, 0, 0]

plt.plot(points[0], points[1], color='blue', label='SOS ROA Estimate')
plt.scatter(100, 100, color='white', label='Simulation-based ROA Estimate')
plt.xlim(-7, 7)
plt.ylim(0, 2*np.pi)
plt.title('Conservativity of SOS ROA Estimate')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('theta (rad)')
plt.savefig('conservativity.png')
