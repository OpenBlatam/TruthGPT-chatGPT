from typing import Callable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

from optimization_core.factories.registry import OPTIMIZERS

@OPTIMIZERS.register("lion")
class Lion(Optimizer):
    r"""
    Implements Lion Algorithm.
    Paper: https://arxiv.org/abs/2302.06675
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-4)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
        weight_decay: weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            exp_avgs = []
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                params.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                
                exp_avgs.append(state["exp_avg"])

            if not params:
                continue
            
            # Lion Update Rule
            # 1. c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            # 2. update = sign(c_t) * lr + wd * p_t * lr
            # 3. m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
            
            for i, p in enumerate(params):
                grad = grads[i]
                exp_avg = exp_avgs[i]

                # Perform stepweight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                # Weight update
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

