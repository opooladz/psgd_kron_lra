import string
import random
import numpy as np
import torch

###############################################################################
# Schedule for preconditioner update probability

def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.
    
    Exponential anneal with a flat start.
    """
    max_prob_ = torch.tensor(max_prob, dtype=torch.float32)
    min_prob_ = torch.tensor(min_prob, dtype=torch.float32)
    decay_ = torch.tensor(decay, dtype=torch.float32)
    flat_start_ = torch.tensor(flat_start, dtype=torch.float32)

    @torch.compile
    def _schedule(n):
        prob = max_prob_ * torch.exp(-decay_ * (n - flat_start_))
        prob.clamp_(min=min_prob_, max=max_prob_)
        return prob

    return _schedule

###############################################################################
# Per–Kronecker–factor LRA initialization and effective preconditioner

def _init_LRA_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None, rank=10):
    """
    For a tensor t with shape (s₁, s₂, …, sₙ), initialize for each dimension an LRA state
    and also compute three einsum strings (exprA, exprGs, exprP) that “encode” the Kronecker product.
    
    Each LRA state is a dict with:
       "U": (s, rank) tensor (or zeros if a diagonal preconditioner is used)
       "V": (s, rank) tensor (or zeros if a diagonal preconditioner is used)
       "d": (s, 1) tensor
    """
    letters = string.ascii_lowercase + string.ascii_uppercase
    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    n = len(shape)
    if n == 0:
        LRA_state = [{"U": None, "V": None, "d": scale * torch.ones_like(t, dtype=dtype)}]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:
        # Distribute scale equally among factors.
        scale_factor = scale ** (1 / n)
        # Decide which factors use a diagonal preconditioner.
        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            dim_diag = [False for _ in shape]

        LRA_state = []
        # For constructing einsum strings.
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, diag_flag) in enumerate(zip(shape, dim_diag)):
            if size == 1 or size > max_size or n < min_ndim_triangular or diag_flag:
                # Use a diagonal preconditioner: U and V are zeros.
                U = torch.zeros(size, rank, dtype=dtype, device=t.device)
                V = torch.zeros(size, rank, dtype=dtype, device=t.device)
                d = scale_factor * torch.ones(size, 1, dtype=dtype, device=t.device)
                LRA_state.append({"U": U, "V": V, "d": d})
                piece1A.append(letters[i])
                piece2A += letters[i]
                piece3A += letters[i]
                piece1 = "".join([(letters[i + n] if j == i else letters[j]) for j in range(n)])
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + n])
                piece1P.append(letters[i + n])
                piece2P.append(letters[i + n])
                piece3P += letters[i + n]
                piece4P += letters[i + n]
            else:
                # Use full triangular factors.
                U = torch.randn(size, rank, dtype=dtype, device=t.device) / np.sqrt(size * (rank + 10))
                V = torch.randn(size, rank, dtype=dtype, device=t.device) / np.sqrt(size * (rank + 10))
                d = scale_factor * torch.ones(size, 1, dtype=dtype, device=t.device)
                LRA_state.append({"U": U, "V": V, "d": d})
                piece1A.append(letters[i] + letters[i + n])
                piece2A += letters[i + n]
                piece3A += letters[i]
                piece1 = "".join([(letters[i + n] if j == i else letters[j]) for j in range(n)])
                piece2 = "".join([(letters[i + 2 * n] if j == i else letters[j]) for j in range(n)])
                exprGs.append(piece1 + "," + piece2 + "->" + letters[i + n])
                a, b, c = letters[i], letters[i + n], letters[i + 2 * n]
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P += c
                piece4P += b
        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    exprGs = tuple(exprGs)
    return LRA_state, (exprA, exprGs, exprP)

def effective_Q(state):
    """
    Given a single LRA state (for one factor), return the effective preconditioner matrix:
         Q_eff = diagflat(d) + U * V^T
    If U and V are None (i.e. a diagonal preconditioner), this returns diagflat(d).
    """
    if state["U"] is None or state["V"] is None:
        return torch.diagflat(state["d"].flatten())
    else:
        return torch.diagflat(state["d"].flatten()) + state["U"].mm(state["V"].t())

###############################################################################
# LRA update math (whitening; always using 2nd–order normalization)

def damped_pair_vg(g, damp=2**(-13)):
    """Return (v, g_damped) for gradient vector g."""
    v = torch.randn_like(g)
    return v, g + damp * torch.mean(torch.abs(g)) * v

def IpUVtmatvec(U, V, x):
    """Compute (I + U*V^T) * x."""
    return x + U.mm(V.t().mm(x))

def update_precond_UVd_math_(U, V, d, v, h, step, tiny):
    """
    Update preconditioner Q = (I + U*V^T)*diag(d) with a (v, h) pair,
    using 2nd–order normalization. In the Kron setting, h (and thus nablaD)
    may be a full square matrix. In that case, we update d only along its diagonal.

    Assumes d is either a 1D tensor of shape (s,) or a column vector of shape (s,1).
    """
    # Ensure v and h are 2D.
    if h.dim() == 1:
        h = h.unsqueeze(1)
    if v.dim() == 1:
        v = v.unsqueeze(1)
    
    # Occasionally balance U and V.
    if torch.rand([]) < 0.01:
        normU = torch.linalg.vector_norm(U)
        normV = torch.linalg.vector_norm(V)
        rho = torch.sqrt((normU / normV) + tiny)
        U.div_(rho)
        V.mul_(rho)
    
    # Ensure d is used as a column vector.
    d_col = d if d.dim() == 2 else d.unsqueeze(1)
    
    Qh = IpUVtmatvec(U, V, d_col * h)
    Ph = d_col * IpUVtmatvec(V, U, Qh)
    
    VtU = V.t().mm(U)
    I = torch.eye(VtU.size(0), dtype=VtU.dtype, device=VtU.device)
    IpVtU = I + VtU
    
    invQtv = v / d_col
    LU, pivots = torch.linalg.lu_factor(IpVtU)
    invQtv = invQtv - V.mm(torch.linalg.lu_solve(LU, pivots, U.t().mm(invQtv), adjoint=True))
    invPv = invQtv - U.mm(torch.linalg.lu_solve(LU, pivots, V.t().mm(invQtv)))
    invPv = invPv / d_col
    
    nablaD = Ph * h - v * invPv
    mu = step * torch.min(torch.rsqrt(Ph*Ph + v*v + tiny) *
                            torch.rsqrt(h*h + invPv*invPv + tiny))
    
    # Instead of updating d with the full nablaD (which would broadcast d to shape [s,s]),
    # we update d only along its diagonal.
    nablaD_diag = torch.diag(nablaD)  # shape: (s,)
    d_update = mu * d_col.squeeze() * nablaD_diag  # update: shape (s,)
    if d.dim() == 2:
        d.sub_(d_update.unsqueeze(1))
    else:
        d.sub_(d_update)
    
    # Update U and V with a symmetric randomized choice.
    a, b = Qh, invQtv
    if torch.rand([]) < 0.5:
        atV = a.t().mm(V)
        btV = b.t().mm(V)
        atVVt = atV.mm(V.t())
        btVVt = btV.mm(V.t())
        mu2 = step / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(atVVt) +
                      torch.linalg.vector_norm(b) * torch.linalg.vector_norm(btVVt) + tiny)
        U.sub_(mu2 * (a.mm(atV.mm(IpVtU)) - b.mm(btV.mm(IpVtU))))
    else:
        atU = a.t().mm(U)
        btU = b.t().mm(U)
        UUta = U.mm(atU.t())
        UUtb = U.mm(btU.t())
        mu2 = step / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(UUta) +
                      torch.linalg.vector_norm(b) * torch.linalg.vector_norm(UUtb) + tiny)
        V.sub_(mu2 * ((a + V.mm(atU.t())).mm(atU) - (b + V.mm(btU.t())).mm(btU)))

def precond_grad_UVd_math(U, V, d, g):
    """
    Precondition gradient g using the LRA state:
         pre_grad = d * (I + V*U^T) (d*g + U*(V^T*(d*g)))
    (This is equivalent to applying Q = diag(d) + U*V^T on g twice.)
    """
    tmp = IpUVtmatvec(U, V, d * g)
    return d * IpUVtmatvec(V, U, tmp)

###############################################################################
# Functions similar to those in the original Kron code

@torch.compile
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))

def _lb(A: torch.Tensor, max_abs: torch.Tensor):
    """Cheap lower bound for the spectral norm of A."""
    A /= max_abs
    a0 = torch.einsum("ij,ij->j", A, A)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("j,kj->k", x, A)
    x = x.norm()
    x *= max_abs
    return x

@torch.compile
def _solve_triangular_right(X: torch.Tensor, A: torch.Tensor):
    """Compute X @ inv(A)."""
    orig_dtype = A.dtype
    return (
        torch.linalg.solve_triangular(
            A.float(),
            X.reshape(-1, X.size(-1)).float(),
            upper=True,
            left=False,
            unitriangular=False,
        )
        .to(dtype=orig_dtype)
        .reshape_as(X)
    )

@torch.compile
def _calc_A_and_conjB(exprA, G, Q_list):
    """
    Calculate A and a conjugate term (conjB) using the einsum expression exprA,
    given effective preconditioners Q_list (one per factor) and gradient G.
    """
    order = G.dim()
    V_rand = torch.randn_like(G)
    eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=G.dtype, device=G.device)
    G = G + eps.sqrt() * G.abs().mean() * V_rand
    conjB = V_rand.permute(*list(range(1, order)) + [0])
    for i, q in enumerate(Q_list):
        if q.dim() < 2:
            conjB = conjB / q
        else:
            conjB = _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    A = torch.einsum(exprA, *Q_list, G)
    return A, conjB

@torch.compile
def _update_precond_LRA(LRA_state, exprs, G, step, tiny):
    """
    Update each factor’s LRA state given the gradient G.
    
    LRA_state is a list (one per factor), and exprs is a tuple (exprA, exprGs, exprP).
    For each factor i, we compute:
       update_vec = einsum(exprGs[i], A, A) - einsum(exprGs[i], conjB, conjB)
    and then update the i-th LRA state via update_precond_UVd_math_.
    """
    exprA, exprGs, _ = exprs
    # Compute effective Q for each factor.
    Q_list = [effective_Q(s) for s in LRA_state]
    A, conjB = _calc_A_and_conjB(exprA, G, Q_list)
    for i, exprG in enumerate(exprGs):
        term1 = torch.einsum(exprG, A, A)
        term2 = torch.einsum(exprG, conjB, conjB)
        update_vec = term1 - term2
        # Update the i-th factor’s LRA state.
        state_i = LRA_state[i]
        v_damp, h_damp = damped_pair_vg(update_vec)
        update_precond_UVd_math_(state_i["U"], state_i["V"], state_i["d"],
                                 v_damp, h_damp, step, tiny)

@torch.compile
def _precond_grad(exprP, LRA_state, G):
    """
    Precondition the gradient G using the effective preconditioners from each factor.
    """
    Q_list = [effective_Q(s) for s in LRA_state]
    return torch.einsum(exprP, *Q_list, *Q_list, G)

@torch.compile
def _clip_update_rms(g):
    g.mul_(
        torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            1.1 / (g.square().mean().sqrt().add(1e-12)),
        )
    )

###############################################################################
# KronLRA Optimizer

class KronLRA(torch.optim.Optimizer):
    """
    Implements a Kroncker–factorized PSGD optimizer where the preconditioner
    for each parameter is built as a Kronecker product over its dimensions.
    For each factor (dimension) we use a low–rank approximation (LRA) update
    (whitening style, always using 2nd–order normalization).
    
    The overall preconditioning is computed via:
       pre_grad = einsum(exprP, *[effective_Q(s) for s in LRA_state], *[effective_Q(s) for s in LRA_state], G)
    where G is the debiased momentum.
    """
    def __init__(
        self,
        params,
        lr=0.0003,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        precond_lr=0.1,
        precond_init_scale=1.0,
        rank_LRA=10,
        mu_dtype=None,
        precond_dtype=None,
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()
        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            rank_LRA=rank_LRA,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
        )
        super(KronLRA, self).__init__(params, defaults)
        self._prob_step = torch.tensor(0, dtype=torch.int32)
        self._update_counter = torch.tensor(0, dtype=torch.int32)
        self.rng = random.Random(42)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mu_dtype = group.get("mu_dtype")
            precond_dtype = group.get("precond_dtype", torch.float32)
            if precond_dtype is None:
                precond_dtype = torch.float32
            momentum_into_precond_update = group.get("momentum_into_precond_update", True)
            precond_lr = torch.tensor(group["precond_lr"], dtype=precond_dtype)
            tiny_tensor = torch.tensor(torch.finfo(precond_dtype).tiny, dtype=precond_dtype)

            # Determine update probability (if callable, use current step)
            update_prob = group["preconditioner_update_probability"]
            if callable(update_prob):
                update_prob = update_prob(self._prob_step.to(dtype=torch.float32))
            self._update_counter += 1
            do_update = self._update_counter >= 1 / update_prob
            if do_update:
                self._update_counter = torch.tensor(0, dtype=torch.int32)
            self._prob_step += 1

            # Optionally balance preconditioners roughly every 100 updates.
            balance = self.rng.random() < 0.01 and do_update

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p, dtype=mu_dtype or p.dtype)
                    # Initialize per–Kronecker–factor LRA state and einsum expressions.
                    state["LRA"], state["exprs"] = _init_LRA_exprs(
                        p,
                        group["precond_init_scale"],
                        group["max_size_triangular"],
                        group["min_ndim_triangular"],
                        group["memory_save_mode"],
                        dtype=precond_dtype,
                        rank=group["rank_LRA"],
                    )
                state["step"] += 1
                beta = group["b1"]
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                if mu_dtype is not None:
                    momentum_buffer.copy_(momentum_buffer.to(dtype=mu_dtype))
                debiased_momentum = momentum_buffer / (1 - beta ** state["step"])
                debiased_momentum = debiased_momentum.to(dtype=precond_dtype)

                if p.dim() > 1 and balance:
                    _balance_Q([effective_Q(s) for s in state["LRA"]])

                # Update the per–factor preconditioners.
                if do_update:
                    # Use momentum into precond update if requested.
                    grad_for_precond = debiased_momentum if momentum_into_precond_update else grad.to(dtype=precond_dtype)
                    _update_precond_LRA(state["LRA"], state["exprs"], grad_for_precond, precond_lr, tiny_tensor)

                # Precondition the gradient.
                pre_grad = _precond_grad(state["exprs"][-1], state["LRA"], debiased_momentum)
                _clip_update_rms(pre_grad)
                if group["weight_decay"] != 0 and p.dim() >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])
                p.add_(pre_grad.to(dtype=p.dtype), alpha=-group["lr"])
        return loss
