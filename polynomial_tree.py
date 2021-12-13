from dircol_problem import DircolProblem
from ellipsoid_dircol_problem import EllipsoidDircolProblem
import numpy as np
from scipy.linalg import solve_continuous_are, expm
from scipy.integrate import solve_ivp
import bisect
from pydrake.all import MathematicalProgram, Solve, Polynomial, Variables, Jacobian
from pydrake.symbolic import TaylorExpand, cos, sin, Expression
import matplotlib.pyplot as plt
import cvxpy as cp
from multiprocessing import Process

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
    return False

  success = not (problem.status == 'infeasible' or problem.status == 'unbounded')

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

def tvlqr(xs, us, ts, dynamics_deriv, Q, R, Qf):
  nsdot = NSdot(ts, xs, us, Q, R, dynamics_deriv)
  ST = Qf.flatten()
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
    return np.dot(diff, self.S@diff) <= self.rho, 0
    
  def get_u(self, x, t):
    return self.u0 - self.K@(x - self.x0)

class PolynomialPolicy(object):
  def __init__(self, x0, u, V, rho, err_vars, S):
    self.x0 = np.copy(x0)
    self.u = u
    self.V = V
    self.rho = rho
    self.err_vars = err_vars
    self.S = np.copy(S)

  def infinite_horizon(self):
    return True

  def get_x0(self, t):
    return np.copy(self.x0)

  def set_rho(self, rho):
    self.rho = rho

  def get_rho(self, t):
    return self.rho

  def get_S(self, t):
    return np.copy(self.S)

  def in_roa(self, x):
    if self.rho is None:
      print('Querying region of attraction without setting it first')
      return False

    diff = x - self.x0
    return self.V.Substitute({v: d for v, d in zip(self.err_vars, diff)}).Evaluate() <= self.rho, 0
    
  def get_u(self, x, t):
    diff = x - self.x0
    return np.array([u.Substitute({v: d for v, d in zip(self.err_vars, diff)}).Evaluate() for u in self.u])

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

  # TODO: check if x is between ellipsoids
  def in_roa(self, x):
    if self.rhos is None:
      print('Querying region of attraction without setting it first')
      return False

    # Check roa of all funnels
    for t, S, rho, x0 in zip(self.ts, self.Ss, self.rhos, self.xs):
      diff = x - x0
      if np.dot(diff, S@diff) <= rho:
        return True, t

    return False, 0

  def get_u(self, x, t):
    K = self.get_item(self.Ks, t)
    x0 = self.get_item(self.xs, t)
    u0 = self.get_item(self.us, t)
    return u0 - K@(x - x0)

class TVPolynomialPolicy(object):
  def __init__(self, xs, us, ts, Vs, rhos, errs, Ss, Sdots):
    self.xs = [np.copy(x) for x in xs]
    self.us = us
    self.ts = [t for t in ts]
    self.Vs = Vs
    self.rhos = rhos
    self.errs = errs
    self.Ss = [np.copy(S) for S in Ss]
    self.Sdots = [np.copy(Sdot) for Sdot in Sdots]

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

  def get_x0(self, t):
    return self.get_item(self.xs, t)

  def get_Sdot(self, t):
    return self.get_item(self.Sdots, t)

  def get_S(self, t):
    return self.get_item(self.Ss, t)

  def set_rhos(self, rhos):
    self.rhos = [rho for rho in rhos]

  def get_rho(self, t):
    return self.get_item(self.rhos, t)

  def in_roa(self, x):
    if self.rhos is None:
      print('Querying region of attraction without setting it first')
      return False

    # Check roa of all funnels
    for t, V, rho, err_vars, x0 in zip(self.ts, self.Vs, self.rhos, self.errs, self.xs):
      if V is None:
        continue
      diff = x - x0
      if V.Substitute({v: di for v, di in zip(err_vars, diff)}).Evaluate() <= rho:
        return True, t

    return False, 0

  def get_u(self, x, t):
    diff = x - self.get_x0(t)
    step = bisect.bisect_right(self.ts, t) - 1
    u = self.us[step]
    err_vars = self.errs[step]
    return np.array([ui.Substitute({v: d for v, d in zip(err_vars, diff)}).Evaluate() for ui in u])

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
  def __init__(self, policy, parent, parent_t):
    self.policy = policy
    self.parent = parent
    self.parent_t = parent_t

class PolynomialTree(object):
  def __init__(self, xmax, symbolic_dynamics_u, dynamics, dynamics_deriv, nx, nu, xgoal, ugoal, ulb, uub, dt, branch_horizon):
    self.xmax = xmax
    self.symbolic_dynamics_u = symbolic_dynamics_u
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

  def trace(self, xinit, xlabel, ylabel, fname):
    xs = []
    us = []

    # For funnel plotting. Each policy gets a list within each list
    Slists = []
    rholists = []
    x0lists = []

    x = np.copy(xinit)
    node = None
    t = 0
    for n in self.nodes:
      in_roa, t = n.policy.in_roa(x)
      if in_roa:
        node = n
        break
    
    if node is None:
      print('This initial condition is not in the ROA of the tree')
      quit()
  
    while node is not None:
      Slists.append([])
      rholists.append([])
      x0lists.append([])
      if node.parent is None:
        # LTI
        x0 = node.policy.get_x0(0)
        Slists[-1].append(node.policy.get_S(0))
        rholists[-1].append(node.policy.get_rho(0))
        x0lists[-1].append(x0)
        while np.linalg.norm(x - x0) > 0.01:
          u = node.policy.get_u(x, 0)
          xs.append(x)
          us.append(u)
          x = integrate(x, u, self.dynamics, self.dt)
      else:
        # LTV
        while t < node.policy.ts[-1]:
          Slists[-1].append(node.policy.get_S(t))
          rholists[-1].append(node.policy.get_rho(t))
          x0lists[-1].append(node.policy.get_x0(t))

          u = node.policy.get_u(x, t)
          xs.append(x)
          us.append(u)
          x = integrate(x, u, self.dynamics, self.dt)
          t += self.dt

      t = node.parent_t
      node = node.parent

    # Plotting
    plt.figure()
    plt.plot([x[0] for x in xs], [x[1] for x in xs], color='r', label='Trajectory')

    num_funnels = len(Slists)
    if num_funnels > 1:
      colors = [[0, f/(num_funnels - 1), 1 - f/(num_funnels - 1)] for f in range(num_funnels)]
    else:
      colors = [[0, 1, 0] for f in range(num_funnels)]

    colors[-1] = [0, 0, 0]

    for color, Ss, rhos, x0s, f in zip(colors, Slists, rholists, x0lists, list(range(num_funnels))):
      num_pts = len(Ss)
      for S, rho, x0, p in zip(Ss, rhos, x0s, list(range(num_pts))):
        # Project S to 2D using Schur complement
        J = S[:2, :2]
        L = S[2:, :2]
        K = S[2:, 2:]
        Sp = J - L.transpose()@np.linalg.solve(K, L)

        theta = np.linspace(0, 2*np.pi, 100)
        points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
        L = np.linalg.cholesky(Sp)
        points = np.linalg.solve(L.transpose(), points)

        for i in range(2):
          points[i] += x0[i]

        if f == num_funnels - 1 and p == num_pts - 1:
          plt.plot(points[0], points[1], color=color, linewidth=6, label='LTI Controller')
        else:
          plt.plot(points[0], points[1], color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Funnel Trace Starting at ' + str(xinit))
    plt.legend()
    plt.savefig(fname)
    plt.show()

    return xs, us

  def plot_all_funnels(self, xlabel, ylabel, fname):
    plt.figure()
    num_funnels = len(self.nodes)
    if num_funnels > 1:
      colors = [[0, 1 - f/(num_funnels - 1), f/(num_funnels - 1)] for f in range(num_funnels)]
    else:
      colors = [[0, 1, 0] for f in range(num_funnels)]

    colors[0] = [0, 0, 0]

    for color, node in zip(colors, self.nodes):
      if node.policy.infinite_horizon():
        Ss = [node.policy.S]
        rhos = [node.policy.rho]
        x0s = [node.policy.x0]
      else:
        Ss = node.policy.Ss
        rhos = node.policy.rhos
        x0s = node.policy.xs

      for S, rho, x0 in zip(Ss, rhos, x0s):
        # Project S to 2D using Schur complement
        J = S[:2, :2]
        L = S[2:, :2]
        K = S[2:, 2:]
        Sp = J - L.transpose()@np.linalg.solve(K, L)

        theta = np.linspace(0, 2*np.pi, 100)
        points = np.sqrt(rho)*np.stack((np.cos(theta), np.sin(theta)), 0)
        L = np.linalg.cholesky(Sp)
        points = np.linalg.solve(L.transpose(), points)

        for i in range(2):
          points[i] += x0[i]

        if node.policy.infinite_horizon():
          plt.plot(points[0], points[1], color=color, label='LTI Controller')
        else:
          plt.plot(points[0], points[1], color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Funnel Library')
    plt.savefig(fname)
    plt.show()

  def find_roa_and_controller(self, lqr_policy, t, terminal, tnext=None, rhonext=None):
    prog = MathematicalProgram()
    xerr = prog.NewIndeterminates(self.nx, 'xerr')
    upolys = np.array([prog.NewFreePolynomial(Variables(xerr), 2) for i in range(self.nu)])
    u = np.array([poly.ToExpression() for poly in upolys])

    x0 = lqr_policy.get_x0(t)
    u0 = lqr_policy.get_u0(t)
    if terminal:
      x0dot = np.zeros_like(x0)
    else:
      x0dot = self.dynamics(x0, u0)

    xerrdot = self.symbolic_dynamics_u(xerr + x0, t, u) - x0dot

    S = lqr_policy.get_S(t)

    if terminal:
      Sdot = np.zeros_like(S)
    else:
      Sdot = lqr_policy.get_Sdot(t)

    for i in range(self.nx):
      xerrdot[i] = TaylorExpand(xerrdot[i], {var: 0 for var in xerr}, 3) 

    V = np.dot(xerr, S@xerr)
    Vdot = 2*np.dot(xerr, S@xerrdot) + np.dot(xerr, Sdot@xerr)

    if terminal:
      la = prog.NewSosPolynomial(Variables(xerr), 2)[0].ToExpression()
    else:
      la = prog.NewFreePolynomial(Variables(xerr), 2).ToExpression()

    # For actuator limits
    ula_upper = [prog.NewSosPolynomial(Variables(xerr), 2)[0].ToExpression() for ui in u]
    ula_lower = [prog.NewSosPolynomial(Variables(xerr), 2)[0].ToExpression() for ui in u]

    # Line search
    lower = 0
    if rhonext is None:
      rho = 25
    else:
      rho = rhonext*2

    upper = rho*2

    max_improve = 10
    rho_min = 1e-3
    i = 0
    bestu = None
    while rho > rho_min:
      print('ROA line search iteration %d, testing rho = %f' %(i, rho))
      if i > max_improve and lower != 0:
        break

      if terminal:
        rhodot = 0
      else:
        rhodot = (rhonext - rho)/(tnext - t)

      prog_clone = prog.Clone()

      if terminal:
        prog_clone.AddSosConstraint(rhodot - Vdot - la*(rho - V) - 0.001*xerr@xerr)
      else:
        prog_clone.AddSosConstraint(rhodot - Vdot - la*(rho - V))

      # Actuator limits
      for ui, ulai_upper, ulai_lower, lb, ub in zip(u, ula_upper, ula_lower, self.ulb, self.uub):
        prog_clone.AddSosConstraint(ub - ui - ulai_upper*(rho - V))
        prog_clone.AddSosConstraint(ui - lb - ulai_lower*(rho - V))

      result = Solve(prog_clone)

      if result.is_success():
        lower = rho
        rho = (rho + upper)/2
        bestu = np.array([result.GetSolution(poly).ToExpression() for poly in upolys])
      else:
        upper = rho
        rho = (rho + lower)/2

      i += 1

    if lower == 0:
      print('No region of attraction')

    rho = lower
    print('Finished ROA line search with rho = %f' %(rho))

    return rho, bestu, V, Variables(xerr)

  def nearest_neighbor_euclidean(self, x):
    # Euclidean distance version
    min_dist = np.inf
    min_idx = 0
    min_t = 0
    for n, node in enumerate(self.nodes):
      if node.policy.infinite_horizon():
        policy_ts = [0]
      else:
        policy_ts = node.policy.ts[:-1]

      for t in policy_ts:
        x0 = node.policy.get_x0(t)
        dist = np.linalg.norm(x - x0)
        if dist < min_dist:
          min_dist = dist
          min_idx = n
          min_t = t

    return self.nodes[min_idx], min_t

  def nearest_neighbor_aqr(self, x, R):
    uf = np.zeros(self.nu)
    minJ = np.inf
    min_idx = 0
    min_t = 0
    for n, node in enumerate(self.nodes):
      # Affine quadratic regulator to get from this node to x. Linearize about x (I guess that makes sense, since it's the goal
      if node.policy.infinite_horizon():
        policy_ts = [0]
      else:
        policy_ts = node.policy.ts[:-1]

      for t in policy_ts:
        x0 = node.policy.get_x0(t)

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
          min_t = t

    return self.nodes[min_idx], min_t


  def build_tree(self):
    # Infinite-horizon LQR at the goal
    A, B = self.dynamics_deriv(self.xgoal, self.ugoal)
    Q = np.eye(self.xgoal.shape[0])
    R = np.eye(self.ugoal.shape[0])
    S = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.transpose()@S)

    lqr_policy = LQRPolicy(self.xgoal, self.ugoal, S, K)

    # Find roa
    rho, u, V, err_vars = self.find_roa_and_controller(lqr_policy, 0, True)
    if rho == 0:
      print('Failed to verify terminal controller')
      quit()

    self.nodes.append(Node(PolynomialPolicy(self.xgoal, u, V, rho, err_vars, S), None, 0))

    max_nodes = 200
    dircol_fails = 0
    for n in range(max_nodes):
      # Randomly sample points until we find one not in a funnel
      in_funnel = True
      max_samples = 1000
      for sample in range(max_samples):
        # Randomly sample a point
        xsample = np.random.uniform(self.xgoal - self.xmax, self.xgoal + self.xmax)

        # Check if it's in a funnel already
        in_funnel = False
        for node in self.nodes:
          in_roa, _ = node.policy.in_roa(xsample)
          if in_roa:
            in_funnel = True

        if not in_funnel:
          break

      if in_funnel:
        print('Space has been probabilistically covered wihth %d funnels. Direct collocation failed %d times' %(len(self.nodes), dircol_fails))
        return

      # Find closest node in tree to this new node using the affine quadratic regulator
      nearest_node, t = self.nearest_neighbor_aqr(xsample, R)
      xnext = nearest_node.policy.get_x0(t)
      Snext = nearest_node.policy.get_S(t)
      rhonext = nearest_node.policy.get_rho(t)

      problem = DircolProblem(Q, R, self.branch_horizon, self.dt, self.dynamics, self.dynamics_deriv, self.nx, self.nu, xsample, xnext, self.ulb, self.uub)
      xs, us, solved = problem.solve()
      if not solved:
        print('Direct collocation failed. %d funnels so far' %(len(self.nodes)))
        '''
        print(xsample)
        print(xnext)

        nearest_node, t = self.nearest_neighbor_aqr(xsample, R)
        xnext = nearest_node.policy.get_x0(t)
        problem = DircolProblem(Q, R, self.branch_horizon, self.dt, self.dynamics, self.dynamics_deriv, self.nx, self.nu, xsample, xnext, self.ulb, self.uub)
        xs, us, solved = problem.solve()
        print(xnext)
        print(solved)
        print(xs)
        print(us)
        quit()
        '''

        dircol_fails += 1
        continue

      # Run TVLQR to stabilize the nominal trajectory
      ts = [step*self.dt for step in range(self.branch_horizon + 1)]
      Ss, Ks, Sdots = tvlqr(xs, us, ts, self.dynamics_deriv, Q, R, Snext)

      rho = rhonext

      lqr_policy = TVLQRPolicy(xs, us, ts, Ss, Ks, Sdots) 

      # Run SOS optimization to get the funnel size at each time step
      rhos = [rho]
      us = [None]
      Vs = [None]
      errs = [None]
      for rstep, t in enumerate(reversed(ts[:-1])):
        step = len(ts) - 2 - rstep
        tnext = ts[step + 1]
        rhonext = rhos[-1]
        rho, u, V, err_vars = self.find_roa_and_controller(lqr_policy, t, False, tnext, rhonext)
        if rho == 0:
          break
        rhos.append(rho)
        us.append(u)
        Vs.append(V)
        errs.append(err_vars)

      if rho == 0:
        continue

      rhos = list(reversed(rhos))
      us = list(reversed(us))
      Vs = list(reversed(Vs))
      errs = list(reversed(errs))
      self.nodes.append(Node(TVPolynomialPolicy(xs, us, ts, Vs, rhos, errs, Ss, Sdots), nearest_node, t))

    print('Created %d funnels, but did not cover space. Direct collocation failed %d times' %(len(self.nodes), dircol_fails))
