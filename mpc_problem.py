import numpy as np
import torch
import casadi as cs
import l4casadi as l4c
from dataclasses import dataclass
from typing import Dict, Callable
from abc import ABC, abstractmethod

class SquaredCritic(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, *args, **kwargs):
        out = self.base_model(*args, **kwargs)
        return out**2

@dataclass
class SymbolicMPCProblem:
    N: int
    h: float
    input_delay: float

    ocp_x: cs.MX
    ocp_u: cs.MX
    ocp_p: cs.MX
    
    dynamics_fun: cs.Function  
    lbx_vec: np.ndarray
    ubx_vec: np.ndarray
    ocp_x_slacks: Dict[int, float] 
    lbu_vec: np.ndarray
    ubu_vec: np.ndarray

    # Stage cost
    stage_cost_fun: cs.Function = None
    # For learned neural stage cost:
    ocp_x_to_stage_state_fun: cs.Function = None
    stage_cost_params: cs.MX = None
    get_stage_cost_params_fun: Callable = None

    # Nonlinear constraints:
    lbg_vec: np.ndarray = None
    ubg_vec: np.ndarray = None
    g_fun: cs.Function = None
    ocp_g_slacks: Dict[int, float] = None
    
    # Terminal cost
    terminal_cost_fun: cs.Function = None
    # For learned neural terminal cost:
    ocp_x_to_terminal_state_fun: cs.Function = None
    
    # Terminal nonlinear constraints:
    terminal_lbg_vec: np.ndarray = None
    terminal_ubg_vec: np.ndarray = None
    terminal_g_fun: cs.Function = None
    ocp_t_g_slacks: Dict[int, float] = None

    _stage_l4c_model = None
    has_stage_neural_cost = False

    _terminal_l4c_model = None
    has_terminal_neural_cost = False
    terminal_model_external_shared_lib_dir: str = None
    terminal_model_external_shared_lib_name: str = None

    def add_stage_neural_cost(self, model: torch.nn.Module, model_state: cs.MX):
        self.has_stage_neural_cost = True

        # Casadi complains about vertcat(MX, MX) inputs to a MXfunction, ugly fix is to use a temprorary MX input here:
        tmp_neural_cost_inputs = cs.MX.sym("tmp_s", model_state.size())
        # In the solver we can use a function to go from ocp_x to model_state:
        self.ocp_x_to_stage_state_fun = cs.Function("ocp_x_to_stage_state_fun", [self.ocp_x, self.ocp_u, self.ocp_p], [model_state])

        squared_model = SquaredCritic(model)
        self._stage_l4c_model = l4c.realtime.RealTimeL4CasADi(squared_model, approximation_order=2)
        sym_cost_output = self._stage_l4c_model(tmp_neural_cost_inputs)

        self.stage_cost_params = self._stage_l4c_model.get_sym_params()
        if self.stage_cost_fun is not None:
            output = sym_cost_output + self.stage_cost_fun(self.ocp_x, self.ocp_u, self.ocp_p)
        else:
            output = sym_cost_output
        self.stage_cost_fun = cs.Function("l", [tmp_neural_cost_inputs, self.ocp_x, self.ocp_u, self.stage_cost_params, self.ocp_p], [output])
        self.get_stage_cost_params_fun = self._stage_l4c_model.get_params

    def add_terminal_neural_cost(self, model: torch.nn.Module, model_state: cs.MX):
        assert self.terminal_cost_fun is None, "Trying to assign a terminal neural cost to a problem with an existing terminal cost"
        self.has_terminal_neural_cost = True

        # Casadi complains about vertcat(MX, MX) inputs to a MXfunction, ugly fix is to use a temprorary MX input here:
        tmp_neural_cost_inputs = cs.MX.sym("tmp_t", model_state.size()).T
        # In the solver we can use a function to go from ocp_x to model_state:
        self.ocp_x_to_terminal_state_fun = cs.Function("ocp_x_to_terminal_state_fun", [self.ocp_x, self.ocp_p], [model_state])

        squared_model = SquaredCritic(model)
        self._terminal_l4c_model = l4c.l4casadi.L4CasADi(squared_model, device="cpu")
        sym_cost_output = self._terminal_l4c_model(tmp_neural_cost_inputs)
        self.terminal_cost_fun = cs.Function("l_t", [tmp_neural_cost_inputs, self.ocp_p], [sym_cost_output])
        self.terminal_model_external_shared_lib_dir = self._terminal_l4c_model.shared_lib_dir
        self.terminal_model_external_shared_lib_name = self._terminal_l4c_model.name


class SymbolicMPCSolver(ABC):
    @abstractmethod
    def solve():
        pass
