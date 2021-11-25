from dircol_problem import DircolProblem
from ellipsoid_dircol_problem import EllipsoidDircolProblem
import numpy as np
from scipy.linalg import solve_continuous_are, expm
from scipy.integrate import solve_ivp
import bisect
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
from pydrake.symbolic import TaylorExpand, cos, sin
import matplotlib.pyplot as plt
import cvxpy as cp

def check_ellipse_containment(S1, S2, c1, c2, rho1, rho2):
  nx = c1.shape[0]

  la = cp.Variable(1)
  M1 = np.zeros((nx + 1, nx + 1))
  M2 = np.zeros((nx + 1, nx + 1))
  M1[:nx, :nx] = S1
  M1[nx, :nx] = -np.dot(c1, S1)
  M1[:nx, nx] = -np.dot(c1, S1)
  M1[nx, nx] = np.dot(c1, S1@c1) - rho1
  M2[:nx, :nx] = S2
  M2[nx, :nx] = -np.dot(c2, S2)
  M2[:nx, nx] = -np.dot(c2, S2)
  M2[nx, nx] = np.dot(c2, S2@c2) - rho2
  problem = cp.Problem(cp.Maximize(1), [cp.multiply(la, M1) - M2 >> 0, la >= 0])

  try:
    problem.solve()
  except Exception as e:
    print(e)
    print('Solver failed')
    quit()

  success = not (problem.status == 'infeasible' or problem.status == 'unbounded')

  # TODO: either this is wrong, or the plotting is wrong. The optimizaiton is clearly
  # finding a lambda that ensures the constraint

  return success

class Pdot(object):
  def __init__(self, A, B, R):
    self.A = A
    self.B = B
    self.R = R
    self.nx = A.shape[0]

  def __call__(self, t, P):
    P = P.reshape(self.nx, self.nx)
    Pdot = self.A@P + P@self.A.transpose() + self.B@np.linalg.solve(self.R, self.B.transpose())
    return Pdot.flatten()

class rdot(object):
  def __init__(self, A, c):
    self.A = A
    self.c = c
    self.nx = A.shape[0]

  def __call__(self, t, r):
    return self.A@r + self.c

class NSdot(object):
  def __init__(self, ts, xs, us, Q, R, dynamics_deriv):
    self.ts = ts
    self.xs = xs
    self.us = us
    self.nx = xs[0].shape[0]
    self.Q = Q
    self.R = R
    self.dynamics_deriv = dynamics_deriv

  def __call__(self, t, S):
    if t < self.ts[0]:
      print('Time out of range')
      quit()
    elif t > self.ts[-1]:
      print('Time out of range')
      quit()

    step = bisect.bisect_right(self.ts, t) - 1
    if step == len(self.ts) - 1:
      x = self.xs[-1]
      u = self.us[-1]
    else:
      alpha = (self.ts[step + 1] - t)/(self.ts[step + 1] - self.ts[step])
      x = alpha*self.xs[step] + (1 - alpha)*self.xs[step + 1]
      u = alpha*self.us[step] + (1 - alpha)*self.us[step + 1]
      
    A, B = self.dynamics_deriv(x, u)
    S = np.reshape(S, (self.nx, self.nx))
    NSdot = self.Q - S@B@np.linalg.solve(self.R, B.transpose())@S + S@A + A.transpose()@S
    return NSdot.flatten()

def tvlqr(xs, us, ts, dynamics_deriv, Q, R):
  nsdot = NSdot(ts, xs, us, Q, R, dynamics_deriv)
  ST = Q.flatten()
  t0 = ts[0]
  tf = ts[-1]
  ret = solve_ivp(nsdot, (t0, tf), ST, t_eval=ts)
  nx = xs[0].shape[0]
  Ss = np.flip(ret.y.transpose().reshape(-1, nx, nx), 0)

  derivs = [dynamics_deriv(x, u) for x, u in zip(xs, us)]
  Bs = [B for A, B in derivs]
  Ks = [np.linalg.solve(R, B.transpose()@S) for S, B in zip(Ss, Bs)]
  Sdots = [-nsdot(t, S).reshape(nx, nx) for t, S in zip(ts, Ss)]
  return Ss, Ks, Sdots

class LQRPolicy(object):
  def __init__(self, x0, u0, S, K):
    self.x0 = np.copy(x0)
    self.u0 = np.copy(u0)
    self.S = np.copy(S)
    self.K = np.copy(K)
    self.rho = None

  def infinite_horizon(self):
    return True

  def get_K(self, t):
    return np.copy(self.K)

  def get_x0(self, t):
    return np.copy(self.x0)

  def get_u0(self, t):
    return np.copy(self.u0)

  def get_S(self, t):
    return np.copy(self.S)

  def set_rho(self, rho):
    self.rho = rho

  def get_rho(self, t):
    return self.rho

  def in_roa(self, x):
    if self.rho is None:
      print('Querying region of attraction without setting it first')
      return False

    diff = x - self.x0
    return np.dot(diff, self.S@diff) <= self.rho
    
  def get_u(self, x, t):
    return self.u0 - self.K@(x - self.x0)

class TVLQRPolicy(object):
  def __init__(self, xs, us, ts, Ss, Ks, Sdots):
    self.xs = [np.copy(x) for x in xs]
    self.us = [np.copy(u) for u in us]
    self.ts = [t for t in ts]
    self.Ss = [np.copy(S) for S in Ss]
    self.Ks = [np.copy(K) for K in Ks]
    self.Sdots = [np.copy(Sdot) for Sdot in Sdots]
    self.rhos = None

  def infinite_horizon(self):
    return False

  def get_item(self, arr, t):
    if t < self.ts[0]:
      print('Time out of range')
      quit()
    elif t > self.ts[-1]:
      print('Time out of range')
      quit()

    step = bisect.bisect_right(self.ts, t) - 1
    if step == len(self.ts) - 1:
      item = np.copy(arr[-1])
    else:
      alpha = (self.ts[step + 1] - t)/(self.ts[step + 1] - self.ts[step])
      item = alpha*arr[step] + (1 - alpha)*arr[step + 1]

    return item

  def get_Sdot(self, t):
    return self.get_item(self.Sdots, t)

  def get_K(self, t):
    return self.get_item(self.Ks, t)

  def get_x0(self, t):
    return self.get_item(self.xs, t)

  def get_u0(self, t):
    return self.get_item(self.us, t)

  def get_S(self, t):
    return self.get_item(self.Ss, t)

  def set_rhos(self, rhos):
    self.rhos = [rho for rho in rhos]

  def get_rho(self, t):
    return self.get_item(self.rhos, t)

  # TODO: check if x is in some later part of the funnel, as opposed to
  # just the opening
  def in_roa(self, x):
    if self.rhos is None:
      print('Querying region of attraction without setting it first')
      return False

    diff = x - self.xs[0]
    return np.dot(diff, self.Ss[0]@diff) <= self.rhos[0]

  def get_u(self, x, t):
    K = self.get_item(self.Ks, t)
    x0 = self.get_item(self.xs, t)
    u0 = self.get_item(self.us, t)
    return u0 - K@(x - x0)

def integrate(x, u, dynamics, dt):
  x1 = x
  k1 = dynamics(x1, u)
  x2 = x + dt*k1/2
  k2 = dynamics(x2, u)
  x3 = x + dt*k2/2
  k3 = dynamics(x3, u)
  x4 = x + dt*k3
  k4 = dynamics(x4, u)
  return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

class Node(object):
  def __init__(self, policy, parent):
    self.policy = policy
    self.parent = parent

class LQRTree(object):
  def __init__(self, xmax, symbolic_dynamics, dynamics, dynamics_deriv, nx, nu, xgoal, ugoal, ulb, uub, dt, branch_horizon):
    self.xmax = xmax
    self.symbolic_dynamics = symbolic_dynamics
    self.dynamics = dynamics
    self.dynamics_deriv = dynamics_deriv
    self.nx = nx
    self.nu = nu
    self.xgoal = np.copy(xgoal)
    self.ugoal = np.copy(ugoal)
    self.ulb = np.copy(ulb)
    self.uub = np.copy(uub)
    self.dt = dt
    self.branch_horizon = branch_horizon

    self.nodes = []

  # Projects ellipsoid onto first two state components
  def plot_funnel(self, node):
    if node.policy.infinite_horizon():
      S = node.policy.S

      # Project S using Schur complement
      J = S[:2, :2]
      L = S[2:, :2]
      K = S[2:, 2:]
      Sp = J - L.transpose()@np.linalg.solve(K, L)
      #Sp = K - L@np.linalg.solve(J, L.transpose()) # Plot in velocity space
      
      rho = node.policy.rho

      theta = np.linspace(0, 2*np.pi, 100)
      points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
      L = np.linalg.cholesky(Sp)
      points = np.linalg.solve(L.transpose(), points)

      x0 = node.policy.get_x0(0)
      
      for i in range(2):
        points[i] += x0[i]
      plt.figure()
      plt.plot(points[0], points[1])
      plt.show()
    else:
      plt.figure()
      num_funnels = len(node.policy.ts)
      colors = [[0, f/(num_funnels - 1), 1 - f/(num_funnels - 1)] for f in range(num_funnels)]
      for f, t, S, rho in zip(range(num_funnels), node.policy.ts, node.policy.Ss, node.policy.rhos):
        # Project S using Schur complement
        J = S[:2, :2]
        L = S[2:, :2]
        K = S[2:, 2:]
        Sp = J - L.transpose()@np.linalg.solve(K, L)
        #Sp = K - L@np.linalg.solve(J, L.transpose()) # Plot in velocity space

        theta = np.linspace(0, 2*np.pi, 100)
        points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
        L = np.linalg.cholesky(Sp)
        points = np.linalg.solve(L.transpose(), points)

        x0 = node.policy.get_x0(t)

        for i in range(2):
          points[i] += x0[i]
        plt.plot(points[0], points[1], color=colors[f])

      S = self.nodes[0].policy.S

      # Project S using Schur complement
      J = S[:2, :2]
      L = S[2:, :2]
      K = S[2:, 2:]
      Sp = J - L.transpose()@np.linalg.solve(K, L)
      #Sp = K - L@np.linalg.solve(J, L.transpose()) # Plot in velocity space
      
      rho = self.nodes[0].policy.rho

      theta = np.linspace(0, 2*np.pi, 100)
      points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
      L = np.linalg.cholesky(Sp)
      points = np.linalg.solve(L.transpose(), points)

      x0 = self.nodes[0].policy.get_x0(0)
      
      for i in range(2):
        points[i] += x0[i]
      plt.plot(points[0], points[1], color='r')

      plt.show()

  def trace(self, xinit):
    xs = []
    us = []
    x = np.copy(xinit)
    node = None
    for n in self.nodes:
      if n.policy.in_roa(x):
        node = n
        break
    
    if node is None:
      print('This initial condition is not in the ROA of the tree')
      quit()
  
    while node is not None:
      if node.parent is None:
        # LTI
        x0 = node.policy.get_x0(0)
        while np.linalg.norm(x - x0) > 0.01:
          u = node.policy.get_u(x, 0)
          xs.append(x)
          us.append(u)
          x = integrate(x, u, self.dynamics, self.dt)
      else:
        # LTV
        for step, t in enumerate(node.policy.ts):
          u = node.policy.get_u(x, t)
          xs.append(x)
          us.append(u)
          x = integrate(x, u, self.dynamics, self.dt)

      node = node.parent

    return xs, us

  def find_roa(self, policy, t, terminal, tnext=None, rhonext=None):
    prog = MathematicalProgram()
    xerr = prog.NewIndeterminates(self.nx, 'xerr')
    x0 = policy.get_x0(t)
    u0 = policy.get_u0(t)
    if terminal:
      x0dot = np.zeros_like(x0)
    else:
      x0dot = self.dynamics(x0, u0)

    xerrdot = self.symbolic_dynamics(xerr + x0, t, policy) - x0dot

    S = policy.get_S(t)

    if terminal:
      Sdot = np.zeros_like(S)
    else:
      Sdot = policy.get_Sdot(t)

    for i in range(self.nx):
      xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, 3) 

    V = np.dot(xerr, S@xerr)
    Vdot = 2*np.dot(xerr, S@xerrdot) + np.dot(xerr, Sdot@xerr)

    if terminal:
      la = prog.NewSosPolynomial(Variables(xerr), 2)[0].ToExpression()
    else:
      la = prog.NewFreePolynomial(Variables(xerr), 2).ToExpression()

    # Backtracking line search. TODO: make binary
    rho = 1
    gamma = 0.5
    max_line_search_iter = 10
    for iteration in range(max_line_search_iter):
      if terminal:
        rhodot = 0
      else:
        rhodot = (rhonext - rho)/(tnext - t)

      prog_clone = prog.Clone()
      prog_clone.AddSosConstraint(rhodot - Vdot - la*(rho - V))
      result = Solve(prog_clone)

      if result.is_success():
        break
      else:
        if iteration == max_line_search_iter - 1:
          print('No region of attraction')
          quit()

        rho *= gamma

    return rho

  def nearest_neighbor(self, x, R):
    '''
    # Euclidean distance version
    min_idx = 0
    min_dist = np.inf
    for n, node in enumerate(self.nodes):
      x0 = node.policy.get_x0(0)
      dist = np.linalg.norm(x - x0)
      if dist < min_dist:
        min_idx = n
        min_dist = dist

    return self.nodes[min_idx]
    '''

    uf = np.zeros(self.nu)
    minJ = np.inf
    min_idx = 0
    for n, node in enumerate(self.nodes):
      # Affine quadratic regulator to get from this node to x. Linearize about x (I guess that makes sense, since it's the goal
      x0 = node.policy.get_x0(0)

      A, B = self.dynamics_deriv(x, uf)
      c = self.dynamics(x, uf)

      ts = [self.dt*i for i in range(self.branch_horizon + 1)]
      tf = ts[-1]
      pdot = Pdot(A, B, R)
      ret = solve_ivp(Pdot(A, B, R), (ts[0], ts[-1]), np.zeros((self.nx, self.nx)).flatten(), t_eval=ts)
      Ps = ret.y.transpose().reshape(-1, self.nx, self.nx)
      ret = solve_ivp(rdot(A, c), (ts[0], ts[-1]), np.zeros(self.nx), t_eval=ts)
      rs = ret.y.transpose()
      d = rs[-1] + expm(A*tf)@(x0 - x)
      J = tf + 0.5*np.dot(d, np.linalg.solve(Ps[-1], d)) # TODO: optimize the time
      if J < minJ:
        minJ = J
        min_idx = n

    return self.nodes[min_idx]


  def build_tree(self):
    # Infinite-horizon LQR at the goal
    A, B = self.dynamics_deriv(self.xgoal, self.ugoal)
    Q = np.eye(self.xgoal.shape[0])
    R = np.eye(self.ugoal.shape[0])
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.transpose()@S)

    self.nodes.append(Node(LQRPolicy(self.xgoal, self.ugoal, S, K), None))

    # Find roa
    rho = self.find_roa(self.nodes[-1].policy, 0, True)
    if rho == 0:
      print('Failed to verify terminal controller')
      quit()

    self.nodes[-1].policy.set_rho(rho)

    covered = False
    max_nodes = 200
    for n in range(max_nodes):
      if covered:
        break

      # Randomly sample points until we find one not in a funnel
      in_funnel = True
      max_samples = 100
      for sample in range(max_samples):
        # Randomly sample a point
        xsample = np.random.uniform(self.xgoal -self.xmax, self.xgoal + self.xmax)

        # Check if it's in a funnel already
        in_funnel = False
        for node in self.nodes:
          if node.policy.in_roa(xsample):
            in_funnel = True

        if not in_funnel:
          break

      if in_funnel:
        print('Sampled 100 points and they were all in funnels, but we think the space is not covered')
        quit()

      # Find closest node in tree to this new node using the affine quadratic regulator
      #xsample = np.array([2.41427662, 5.14199858, 0.32134842, -0.0779939]) # TODO: go back to random sampling
      nearest_node = self.nearest_neighbor(xsample, R)
      xnear = nearest_node.policy.get_x0(0)
      Snext = nearest_node.policy.get_S(0)
      rhonext = nearest_node.policy.get_rho(0)

      # Run direct collocation from the new node to the found node. TODO: figure out why ellipsoid version doesn't work
      problem = DircolProblem(Q, R, self.branch_horizon, self.dt, self.dynamics, self.dynamics_deriv, self.nx, self.nu, xsample, xnear, self.ulb, self.uub)
      #problem = EllipsoidDircolProblem(Q, R, self.branch_horizon, self.dt, self.dynamics, self.dynamics_deriv, self.nx, self.nu, xsample, xnear, self.ulb, self.uub, Snext, rhonext)
      xs, us, solved = problem.solve()
      if not solved:
        continue

      # Run TVLQR to stabilize the nominal trajectory
      ts = [step*self.dt for step in range(self.branch_horizon + 1)]
      Ss, Ks, Sdots = tvlqr(xs, us, ts, self.dynamics_deriv, Q, R)

      self.nodes.append(Node(TVLQRPolicy(xs, us, ts, Ss, Ks, Sdots), nearest_node))

      # For the exit of this funnel, we have to find a rho such that the exit of this funnel
      # is contained in the entry of the next funnel. Do backtracking line search
      rho = 1
      x0 = xs[-1]
      S = Ss[-1]

      gamma = 0.5
      max_line_search_iter = 10
      x0next = nearest_node.policy.get_x0(0)
      found = False
      for iteration in range(max_line_search_iter):
        if check_ellipse_containment(S, Snext, x0, x0next, rho, rhonext):
          found = True
          print(rho)
          break
        rho *= gamma

      if not found:
        print('Could not contain the exit of this funnel in the entry of the next funnel')
        quit()

      # TODO: figure out why rho check isn't working
      print(xsample)
      #rho = 0.0625
      rho = 0.0625/2

      # Run SOS optimization to get the funnel size at each time step
      rhos = [rho]
      for rstep, t in enumerate(reversed(ts[:-1])):
        step = len(ts) - 2 - rstep
        tnext = ts[step + 1]
        rhonext = rhos[-1]
        rho = self.find_roa(self.nodes[-1].policy, t, False, tnext, rhonext)
        rhos.append(rho)

      rhos = list(reversed(rhos))
      self.nodes[-1].policy.set_rhos(rhos)

      # Update coverage (TODO, actually implement)
      covered = True
