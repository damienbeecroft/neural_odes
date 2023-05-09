import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
import os
import time
from torch import jit

def adjoint_calculate(t, y, func, method, tol):
    ''' Adapted from torchdiffeq so that I can plot and save values of adjoint. 
        Input is timesteps for eval, state over time (outputted from solver) and solver parameters. '''
    with torch.no_grad():
        adjoint_rtol, adjoint_atol = tol, tol 
        adjoint_method = method
        adjoint_params = tuple(list(func.parameters()))
        grad_y = torch.zeros_like(y)
        grad_y[:, 0] = 1 / len(t) # Loss is mean, so need 1 / len

        ##################################
        #      Set up initial state      #
        ##################################

        # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
        aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
        aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

        ##################################
        #    Set up backward ODE func    #
        ##################################

        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = y_aug[1]
            adj_y = y_aug[2]
            # ignore gradients wrt time and parameters

            with torch.enable_grad():
                t_ = t.detach()
                t = t_.requires_grad_(True)
                y = y.detach().requires_grad_(True)

                # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                # wrt t here means we won't compute that if we don't need it.
                func_eval = func(t, y) # RETURNS TUPLE, NEED TO CONCAT. SEE BELOW.
                cat = [x.reshape(-1) for x in func_eval]
                func_eval = torch.cat(cat)

                # Workaround for PyTorch bug #39784
                _t = torch.as_strided(t, (), ())  # noqa
                _y = torch.as_strided(y, (), ())  # noqa
                _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                    func_eval, (t, y) + adjoint_params, -adj_y,
                    allow_unused=True, retain_graph=True
                )

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
            vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                          for param, vjp_param in zip(adjoint_params, vjp_params)]

            return (vjp_t, func_eval, vjp_y, *vjp_params)

        ##################################
        #       Solve adjoint ODE        #
        ##################################

        # RECORD AUGMENTED SYSTEM OVER ODE.
        record = []
        time_vjps = None
        record.append(aug_state)
        for i in range(len(t) - 1, 0, -1):
            # Run the augmented system backwards in time.
            aug_state = odeint(
                augmented_dynamics, tuple(aug_state),
                t[i - 1:i + 1].flip(0),
                rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method
            )
            aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
            aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
            aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point
            record.append(aug_state)

    return record
