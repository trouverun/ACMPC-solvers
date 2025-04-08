# ACMPC-solvers
This repository provides a flexible interface which can be used to formulate [Actor-Critic MPC](https://arxiv.org/abs/2409.15717) problems using both [CasADi](https://web.casadi.org/) and [Acados](https://docs.acados.org/). 

The "standard" MPC problem is first formulated by instantiating the [SymbolicMPCProblem](./mpc_problem.py) class, as shown here for a unicycle robot:
```python
import casadi as cs
import numpy as np
from mpc_solvers.mpc_problem import SymbolicMPCProblem
from mpc_solvers.acados_sqp_solver import AcadosSQPSolver

fake_inf = 1e7

# States: x, y, theta
x = cs.MX.sym("x")
y = cs.MX.sym("y")
theta = cs.MX.sym("theta")
ocp_x = cs.vertcat(x, y, theta)

# State bounds
lbx_vec = np.array([-fake_inf, -fake_inf, -fake_inf])
ubx_vec = np.array([ fake_inf,  fake_inf,  fake_inf])

# Controls: v (linear velocity), w (angular velocity)
v = cs.MX.sym("v")
w = cs.MX.sym("w")
ocp_u = cs.vertcat(v, w)

# Control bounds
lbu_vec = np.array([-1.0, -1.0])   
ubu_vec = np.array([1.0,  1.0])  

# Parameters: goal pose
x_goal = cs.MX.sym("x_goal")
y_goal = cs.MX.sym("y_goal")
theta_goal = cs.MX.sym("theta_goal")
ocp_p = cs.vertcat(x_goal, y_goal, theta_goal)

# Continuous-time unicycle dynamics
f_expr = cs.vertcat(
    v * cs.cos(theta),   # dx
    v * cs.sin(theta),   # dy
    w                    # dtheta
)
f = cs.Function('f', [ocp_x, ocp_u, ocp_p], [f_expr])

problem = SymbolicMPCProblem(
  N=10,
  h=0.2,
  input_delay=0,
  ocp_x=ocp_x,
  lbx_vec=lbx_vec,
  ubx_vec=ubx_vec,
  ocp_u=ocp_u,
  lbu_vec=lbu_vec,
  ubu_vec=ubu_vec,
  ocp_p=ocp_p,
  dynamics_fun=f
)
```

The neural network cost functions are then added through the function calls:
```python
# These expressions define the mapping from (ocp_x, ocp_u, ocp_p) to the input that is passed on to the neural network,
# here we feed in the position error and sine/cosine encoded heading error (+ controls for the stage cost / Q-function):
hdg_err = theta - theta_goal
critic_stage_state = cs.vertcat(x - x_goal, y - y_goal, cs.sin(hdg_err), cs.cos(hdg_err), v, w)
critic_terminal_state = cs.vertcat(x - x_goal, y - y_goal, cs.sin(hdg_err), cs.cos(hdg_err))

critic_stage_model = torch.load("some_Q_function").eval()
critic_terminal_model = torch.load("some_Value_function").eval()
problem.add_stage_neural_cost(model=critic_stage_model, model_state=critic_stage_state)
problem.add_terminal_neural_cost(model=critic_terminal_model, model_state=critic_terminal_state)
```
which uses [L4CasADi](https://github.com/Tim-Salzmann/l4casadi) to embed an arbitrary torch nn.Module to the symbolic computation graph. The stage cost is added as a second-order Taylor expansion (without the constant term), while the terminal cost is added "as is".

For solving the resulting problem, there are currently two options: 
1. [CasadiCollocationSolver](./casadi_collocation_solver.py) (slow, non-realtime, but more robust)
2. [AcadosSQPSolver](./acados_sqp_solver.py) (fast, real-time, but less robust)
   
Both solvers generate C code from the python problem description, and therefore bypass the slow CasADi VM execution model.

In either case, the solver takes as input the previously constructed problem formulation, and implements a solve method, which solves a single iteration of the MPC problem:
```python
solver = AcadosSQPSolver(problem)
x0 = np.array([5, 5, 0]) # (current) initial state
goal = np.zeros(3)       # goal pose
params = [goal for _ in range(self.problem.N + 1)] # can be a trajectory instead of a fixed goal
sol_x, sol_u, sol = self.solver.solve(x0, params)
```
which returns the state and control horizons sol_x: (N, num_states) and sol_u: (N-1, num_controls).
