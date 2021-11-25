import numpy as np
import cyipopt
from dircol_problem import DircolProblem

class EllipsoidDircolProblem(DircolProblem):
  def __init__(self, Q, R, steps, dt, dynamics, dynamics_deriv, nx, nu, xinit, xterm, ulb, uub, Sterm, rhoterm):
    super().__init__(Q, R, steps, dt, dynamics, dynamics_deriv, nx, nu, xinit, xterm, ulb, uub)
    self.Sterm = Sterm
    self.rhoterm = rhoterm

  def get_num_constraints(self):
    return self.nx*self.steps + 1

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
    cl[-1] = -100
    cu[-1] = self.rhoterm

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
    #lb[-self.vars_per_step:-self.nu] = self.xterm
    #ub[-self.vars_per_step:-self.nu] = self.xterm

    nlp = cyipopt.Problem(
       n=num_decision_vars,
       m=num_constraints,
       problem_obj=self,
       lb=lb,
       ub=ub,
       cl=cl,
       cu=cu,
    )

    soln, info = nlp.solve(warm_start)
    if info['status'] != 0:
      return [], [], False

    xs = []
    us = []
    for step in range(self.steps + 1):
      xs.append(soln[self.vars_per_step*step:self.vars_per_step*step + self.nx])
      us.append(soln[self.vars_per_step*step + self.nx:self.vars_per_step*(step + 1)])

    return xs, us, True

  def constraints(self, x):
    """Returns the constraints."""
    constraint_vals = np.zeros(self.get_num_constraints())
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

    # Terminal state should be within the provided ellipse
    errterm = x[-self.vars_per_step:-self.nu] - self.xterm
    constraint_vals[-1] = np.dot(errterm, self.Sterm@errterm)

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

    errterm = x[-self.vars_per_step:-self.nu] - self.xterm
    vals = np.append(vals, 2*np.dot(errterm, self.Sterm), 0)
 
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

    # Terminal constraint
    last_row = self.get_num_constraints() - 1
    rows.extend([last_row]*self.nx)
    last_state_col = self.get_num_decision_vars() - self.vars_per_step
    cols.extend([last_state_col + i for i in range(self.nx)])
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

    msg = "Objective value at iteration #{:d} is - {:g}"

    print(msg.format(iter_count, obj_value))
