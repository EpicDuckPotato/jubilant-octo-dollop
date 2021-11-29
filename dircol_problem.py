import numpy as np
import cyipopt

class DircolProblem(object):
  def __init__(self, Q, R, steps, dt, dynamics, dynamics_deriv, nx, nu, xinit, xterm, ulb, uub):
    self.Q = Q
    self.R = R
    self.steps = steps
    self.dt = dt
    self.dynamics = dynamics
    self.dynamics_deriv = dynamics_deriv
    self.nx = nx
    self.nu = nu
    self.xinit = np.copy(xinit)
    self.xterm = np.copy(xterm)
    self.ulb = np.copy(ulb)
    self.uub = np.copy(uub)
    self.vars_per_step = self.nx + self.nu

  def solve(self):
    num_decision_vars = self.get_num_decision_vars()
    warm_start = np.zeros(num_decision_vars)

    # Straight line initialization
    xguess = np.linspace(self.xinit, self.xterm, self.steps + 1)
    for i, x in enumerate(xguess):
      warm_start[self.vars_per_step*i:self.vars_per_step*i + self.nx] = x

    num_constraints = self.get_num_constraints()
    cl = np.zeros(num_constraints)
    cu = np.zeros(num_constraints)

    lb = -100*np.ones(num_decision_vars)
    ub = 100*np.ones(num_decision_vars)

    # Actuator constraints
    for step in range(self.steps + 1):
      lb[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)] = self.ulb
      ub[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)] = self.uub

    # Initial constraint
    lb[:self.nx] = self.xinit
    ub[:self.nx] = self.xinit

    # Terminal constraint
    lb[-self.vars_per_step:-self.nu] = self.xterm
    ub[-self.vars_per_step:-self.nu] = self.xterm

    nlp = cyipopt.Problem(
       n=num_decision_vars,
       m=num_constraints,
       problem_obj=self,
       lb=lb,
       ub=ub,
       cl=cl,
       cu=cu,
    )
    nlp.addOption(b'print_level', 0)

    soln, info = nlp.solve(warm_start)
    if info['status'] != 0:
      return [], [], False

    xs = []
    us = []
    for step in range(self.steps + 1):
      xs.append(soln[self.vars_per_step*step:self.vars_per_step*step + self.nx])
      us.append(soln[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)])

    return xs, us, True

  def get_num_decision_vars(self):
    return (self.nx + self.nu)*(self.steps + 1)
    
  def get_num_constraints(self):
    return self.nx*self.steps

  def objective(self, x):
    cost = 0
    for step in range(self.steps + 1):
      u = x[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)]
      cost += 0.5*np.dot(u, self.R@u)
    return cost

  def gradient(self, x):
    """Returns the gradient of the objective with respect to x."""
    grad = np.zeros(x.shape)
    for step in range(self.steps + 1):
      u = x[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)]
      grad[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)] = self.R@u
    return grad

  def constraints(self, x):
    """Returns the constraints."""
    constraint_vals = np.zeros(self.nx*self.steps)
    for step in range(self.steps):
      x0 = x[self.vars_per_step*step:self.vars_per_step*step + self.nx]
      u0 = x[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)]

      x1 = x[self.vars_per_step*(step + 1):self.vars_per_step*(step + 1) + self.nx]
      u1 = x[self.vars_per_step*(step + 1) + self.nx:self.vars_per_step*(step + 2)]

      xdot0 = self.dynamics(x0, u0)
      xdot1 = self.dynamics(x1, u1)

      xcol = 0.5*(x0 + x1) + self.dt/8*(xdot0 - xdot1)
      ucol = (u0 + u1)/2
      
      # Defect should equal zero
      constraint_vals[self.nx*step:self.nx*(step + 1)] = x0 - x1 + self.dt/6*(xdot0 + 4*self.dynamics(xcol, ucol) + xdot1)

    return constraint_vals

  def jacobian(self, x):
    """Returns the Jacobian of the constraints with respect to x."""

    vals = np.zeros(0)
    for step in range(self.steps):
      x0 = x[self.vars_per_step*step:self.vars_per_step*step + self.nx]
      u0 = x[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)]

      x1 = x[self.vars_per_step*(step + 1):self.vars_per_step*(step + 1) + self.nx]
      u1 = x[self.vars_per_step*(step + 1) + self.nx:self.vars_per_step*(step + 2)]

      xdot0 = self.dynamics(x0, u0)
      xdot1 = self.dynamics(x1, u1)

      xcol = 0.5*(x0 + x1) + self.dt/8*(xdot0 - xdot1)
      ucol = (u0 + u1)/2

      A0, B0 = self.dynamics_deriv(x0, u0)
      A1, B1 = self.dynamics_deriv(x1, u1)
      Acol, Bcol = self.dynamics_deriv(xcol, ucol)

      dxcol_dx0 = 0.5*np.eye(self.nx) + self.dt/8*(A0)
      dxcol_dx1 = 0.5*np.eye(self.nx) + self.dt/8*(-A1)

      dxcol_du0 = self.dt/8*B0
      dxcol_du1 = self.dt/8*(-B1)

      dc_dx0 = np.eye(self.nx) + self.dt/6*(A0 + 4*Acol@dxcol_dx0)
      dc_dx1 = -np.eye(self.nx) + self.dt/6*(4*Acol@dxcol_dx1 + A1)
      dc_du0 = self.dt/6*(B0 + 4*(Acol@dxcol_du0 + Bcol*0.5))
      dc_du1 = self.dt/6*(4*(Acol@dxcol_du1 + Bcol*0.5) + B1)

      for row in range(self.nx):
        vals = np.append(vals, dc_dx0[row], 0)
        vals = np.append(vals, dc_du0[row], 0)
        vals = np.append(vals, dc_dx1[row], 0)
        vals = np.append(vals, dc_du1[row], 0)
 
    return vals

  def jacobianstructure(self):
    """Returns the row and column indices for non-zero vales of the
    Jacobian."""
    rows = []
    cols = []
    for step in range(self.steps):
      for row in range(self.nx):
        for col in range(self.vars_per_step):
          rows.append(self.nx*step + row)
          cols.append(self.vars_per_step*step + col)

        for col in range(self.vars_per_step):
          rows.append(self.nx*step + row)
          cols.append(self.vars_per_step*(step + 1) + col)

    return np.array(rows), np.array(cols)

  '''
  def hessianstructure(self):
    """Returns the row and column indices for non-zero vales of the
    Hessian."""
    return None

  def hessian(self, x, lagrange, obj_factor):
    """Returns the non-zero values of the Hessian."""
    return None
  '''

  def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                   d_norm, regularization_size, alpha_du, alpha_pr,
                   ls_trials):
    """Prints information at every Ipopt iteration."""
    return 

    msg = "Objective value at iteration #{:d} is - {:g}"

    print(msg.format(iter_count, obj_value))
