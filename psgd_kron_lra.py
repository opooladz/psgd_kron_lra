import string
import random
import numpy as np
import torch


    
    
def IpUVtmatvec(U, V, x):
    """
    Returns (I + U * Vᵀ) * x.
    All variables are assumed to be either matrices or column vectors.
    """   
    return x + U.mm(V.t().mm(x))

def update_precond_UVd_math_(U, V, d, v, h, step, tiny):
    """
    Update the low-rank preconditioner Q = (I + U * Vᵀ) * diag(d) using the pair (v, h)
    where h approximates the Hessian-vector product.
    
    The state variables U, V, and d are updated in-place.
    All arguments (U, V, d, v, h) are either matrices or column vectors.
    """
    # Optional balancing of U and V
    if torch.rand([]) < 0.01:
        normU = torch.linalg.vector_norm(U)
        normV = torch.linalg.vector_norm(V)
        rho = torch.sqrt(normU / normV)
        U.div_(rho)
        V.mul_(rho)

    Qh = IpUVtmatvec(U, V, d * h)
    Ph = d * IpUVtmatvec(V, U, Qh)
    
    # Solve for invQtv and invPv using LU factorization
    VtU = V.t().mm(U)
    I = torch.eye(VtU.size(0), dtype=VtU.dtype, device=VtU.device)
    IpVtU = I + VtU
    invQtv = v / d
    LU, pivots = torch.linalg.lu_factor(IpVtU)
    invQtv = invQtv - V.mm(torch.linalg.lu_solve(LU, pivots, U.t().mm(invQtv), adjoint=True))
    invPv  = invQtv - U.mm(torch.linalg.lu_solve(LU, pivots, V.t().mm(invQtv)))
    invPv = invPv / d

    # Compute the gradient for updating d
    nablaD = Ph * h - v * invPv
    mu = step * torch.min(torch.rsqrt(Ph * Ph + v * v + tiny) *
                            torch.rsqrt(h * h + invPv * invPv + tiny))

    # Randomly update either U or V (but not both simultaneously)
    a, b = Qh, invQtv
    if torch.rand([]) < 0.5:
        atV = a.t().mm(V)
        btV = b.t().mm(V)
        atVVt = atV.mm(V.t())
        btVVt = btV.mm(V.t())
        mu = step / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(atVVt) +
                        torch.linalg.vector_norm(b) * torch.linalg.vector_norm(btVVt) + tiny)

        U.sub_(mu * (a.mm(atV.mm(IpVtU)) - b.mm(btV.mm(IpVtU))))
    else:
        atU = a.t().mm(U)
        btU = b.t().mm(U)
        UUta = U.mm(atU.t())
        UUtb = U.mm(btU.t())
        mu = step / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(UUta) +
                        torch.linalg.vector_norm(b) * torch.linalg.vector_norm(UUtb) + tiny)

        V.sub_(mu * ((a + V.mm(atU.t())).mm(atU) - (b + V.mm(btU.t())).mm(btU)))

def precond_grad_UVd_math(U, V, d, g):
    """
    Precondition the gradient g with Q = (I + U * Vᵀ) * diag(d).
    All variables are assumed to be either matrices or column vectors.
    """
    g = IpUVtmatvec(U, V, d * g)
    g = d * IpUVtmatvec(V, U, g)
    return g

def damped_pair_vg(g, damp=2**(-13)):
    """
    Instead of return (v, g), it returns pair
        (v, g + sqrt(eps)*mean(abs(g))*v)
    such that the covariance matrix of the modified g is lower bound by
        eps * (mean(abs(g)))**2 * I
    This should damp the preconditioner to encourage numerical stability.
    The default amount of damping is 2**(-13), slightly smaller than sqrt(eps('single')). 
    
    If v is integrated out, let's just use the modified g; 
    If hvp is used, recommend to use L2 regularization to lower bound the Hessian, although this method also works. 

    Please check example
        https://github.com/lixilinx/psgd_torch/blob/master/misc/psgd_with_finite_precision_arithmetic.py
    for the rationale to set default damping level to 2**(-13). 
    """
    v = torch.randn_like(g)
    return (v, g + damp*torch.mean(torch.abs(g))*v)



import torch
import torch.nn as nn
from torch.optim import Optimizer

def group_model_params(model: nn.Module):
    """
    Groups parameters of a model so that for each module, its weight and bias
    are placed together in one parameter group.
    
    Args:
        model (nn.Module): The model whose parameters are to be grouped.
        
    Returns:
        List[Dict]: A list of parameter groups (each a dict with a "params" key).
    """
    groups = []
    seen = set()  # to avoid duplication if parameters are shared
    for module_name, module in model.named_modules():
        group = []
        if hasattr(module, 'weight') and module.weight is not None:
            if id(module.weight) not in seen:
                group.append(module.weight)
                seen.add(id(module.weight))
        if hasattr(module, 'bias') and module.bias is not None:
            if id(module.bias) not in seen:
                group.append(module.bias)
                seen.add(id(module.bias))
        if group:
            groups.append({'params': group})
    
    # Some parameters (if any) might not be captured by the module iteration.
    all_params = list(model.parameters())
    leftover = [p for p in all_params if id(p) not in seen]
    if leftover:
        groups.append({'params': leftover})
    
    return groups


def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))


def _clip_update_rms(g):
    g.mul_(
        torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            1.1 / g.square().mean().sqrt().add(1e-12),
        )
    )

class LRAOptimizer(Optimizer):
    r"""Localized LRA preconditioning optimizer.
    
    If a model (nn.Module) is passed, its parameters are automatically grouped so that
    a module’s weight and bias (if present) are preconditioned together.
    
    The preconditioner is of the form:
    
        Q = (I + U * Vᵀ) * diag(d)
    
    and is updated based on second-order (whitening) derivative information.
    
    Args:
        params (iterable or nn.Module): Either an iterable of parameters (or parameter groups)
            or a full nn.Module. In the latter case, parameters are grouped by module (weight and bias together).
        rank (int, optional): Rank of the approximation (max rank of U or V). Default: 10.
        preconditioner_init_scale (float, optional): Initial scale for the diagonal component d.
            If None, d is set dynamically. Default: None.
        lr (float, optional): Learning rate for the parameters. Default: 0.01.
        lr_preconditioner (float, optional): Learning rate for the preconditioner update. Default: 0.1.
        momentum (float, optional): Momentum factor in [0, 1). Default: 0.
        grad_clip_max_norm (float, optional): Maximum norm for gradient clipping. Default: None.
        preconditioner_update_probability (float, optional): Probability of updating the preconditioner
            at each step. Default: 1.0.
    """
    def __init__(self, params, rank=10, preconditioner_init_scale=None,
                 lr=0.01, lr_preconditioner=0.1, momentum=0.0,
                 grad_clip_max_norm=None, preconditioner_update_probability=1.0):
        # If a full model is passed, group its parameters by module.
        if isinstance(params, nn.Module):
            params = group_model_params(params)
        
        defaults = dict(lr=lr,
                        lr_preconditioner=lr_preconditioner,
                        momentum=momentum,
                        grad_clip_max_norm=grad_clip_max_norm,
                        preconditioner_update_probability=preconditioner_update_probability)
        super(LRAOptimizer, self).__init__(params, defaults)
        self.rank = rank
        self.preconditioner_init_scale = preconditioner_init_scale
        # A tiny constant for numerical stability.
        self.tiny = torch.finfo(torch.float32).tiny

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        For each parameter group, the gradients of all parameters in that group are
        flattened and concatenated into a single vector. A local LRA preconditioner
        (with state U, V, d, and momentum m) is then updated using a damped pair of gradient
        information (via external functions `damped_pair_vg` and `update_precond_UVd_math_`).
        The preconditioned gradient is computed (optionally with momentum and gradient clipping)
        and then used to update the parameters.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            
        Returns:
            The loss value evaluated after the update, if closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Accumulate gradients and related information for this group.
            grad_list = []
            params = []
            shapes = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_list.append(p.grad.view(-1, 1))
                params.append(p)
                shapes.append(p.shape)

            if not grad_list:
                continue

            # Concatenate the gradients into a single column vector.
            grad_vec = torch.cat(grad_list, dim=0)

            # Use (or initialize) a local preconditioner for this group.
            if ("precond_state" not in group) or (torch.rand(1).item() < group["preconditioner_update_probability"]):
                if "precond_state" not in group:
                    total_size = grad_vec.shape[0]
                    device, dtype = grad_vec.device, grad_vec.dtype
                    scaling = (total_size * (self.rank + 10)) ** 0.5
                    U = torch.randn(total_size, self.rank, dtype=dtype, device=device) / scaling
                    V = torch.randn(total_size, self.rank, dtype=dtype, device=device) / scaling
                    d = (torch.ones(total_size, 1, dtype=dtype, device=device) * self.preconditioner_init_scale
                         if self.preconditioner_init_scale is not None else None)
                    precond_state = {"U": U, "V": V, "d": d, "m": None}
                    group["precond_state"] = precond_state
                else:
                    precond_state = group["precond_state"]

                if precond_state["d"] is None:
                    precond_state["d"] = (torch.mean(grad_vec ** 4)) ** (-1 / 8) * torch.ones_like(grad_vec)
                # Compute a damped pair from the gradient.
                # (Assume `damped_pair_vg` is defined elsewhere.)
                v_damped, g_damped = damped_pair_vg(grad_vec)
                # Update the preconditioner.
                # (Assume `update_precond_UVd_math_` is defined elsewhere.)
                update_precond_UVd_math_(precond_state["U"], precond_state["V"],
                                         precond_state["d"],
                                         v_damped, g_damped,
                                         group["lr_preconditioner"], self.tiny)
            else:
                precond_state = group["precond_state"]

            # Precondition the gradient (with optional momentum).
            if group["momentum"] > 0:
                if precond_state["m"] is None:
                    precond_state["m"] = (1 - group["momentum"]) * grad_vec
                else:
                    precond_state["m"].mul_(group["momentum"]).add_(grad_vec, alpha=1 - group["momentum"])
                pre_grad = precond_grad_UVd_math(precond_state["U"], precond_state["V"],
                                                 precond_state["d"], precond_state["m"])
            else:
                precond_state["m"] = None
                pre_grad = precond_grad_UVd_math(precond_state["U"], precond_state["V"],
                                                 precond_state["d"], grad_vec)

            # # Apply gradient clipping if necessary.
            # if group["grad_clip_max_norm"] is None:
            #     effective_lr = group["lr"]
            # else:
            #     grad_norm = pre_grad.norm() + self.tiny
            #     effective_lr = group["lr"] * min(group["grad_clip_max_norm"] / grad_norm, 1.0)

            # clip update RMS
            # _clip_update_rms(pre_grad)

            delta = group["lr"] * pre_grad

            # Update the parameters by slicing delta according to each parameter's size.
            start_idx = 0
            for p, shape in zip(params, shapes):
                numel = p.numel()
                p_delta = delta[start_idx: start_idx + numel].view(shape)
                p.data.sub_(p_delta)
                start_idx += numel

        return loss
