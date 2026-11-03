# This file provide a modified version of the L-BFGD optimizer from Pytorch.
# Main version changes are:
# - A new line search method (quadratic approximation) is added, which is more time efficient linesearch method for our setting
# - A slight modification of the strong Wolfe line search (_cubic interpolatotion is modified to avoid numerical issues) and bracket update strategy madapted


import torch
from torch import Tensor

from torch.optim.optimizer import _to_scalar, Optimizer, ParamsT

__all__ = ['sLBFGS']

def _quadratic_interpolate(t0, f0, g0, t1, f1, decrease, increase):
    """
    Quadratic interpolation using:
        f(t0), f'(t0), f(t1)

    Returns minimizer of quadratic model.
    """

    dt = t1 - t0
    denom = 2.0 * (f1 - f0 - g0 * dt)
    t_next = t0 - g0 * dt * dt / denom

    # Add some bound to avoid numerical issues
    if t_next < 0 or torch.isnan(t_next):
        print(f"sLBGS: Quadratic interpolation failed (<0: {t_next < 0}, nan: {torch.isnan(t_next)}, >10  {t_next > 10 * t1}),", end="")
        t_next = t1 * decrease
        print(f" fallback to {t_next} step length...")

    if t_next > 3 * t1:
        print(f"sLBGS: Quadratic interpolation is unstable {t_next}),", end="")
        t_next = t1 * increase
        print(f" fallback to {t_next} step length...")

    return t_next


@torch.no_grad()
def _quadratic_linesearch(obj_func, x, t, d, f, gtd, max_ls=10, increase=1.5, decrease=0.2, min_step=1e-16):
    """
    Create a quadratic model of the function and minimize it. No need to compute the gradient during line search.

    Args:
    - obj_func: a function that takes (x, t, d) and returns f(x + t d) (no gradient needed)
    - x: current point
    - t: initial step size
    - d: search direction
    - f: f(x)
    - gtd: directional derivative at x (g(x) * d)
    - max_ls: maximum number of line search iterations. Should be at least 2 to perform interpolation.
    - increase: factor to increase t for the first try
    - decrease: factor to decrease t if the first try fails
    - min_step: minimum step size to consider

    Outputs:
    - f_new: f(x + t d) at the chosen step size
    - g_new: is torch.zeros_like(d) since we don't compute the gradient during line search
    - t: the chosen step size
    - ls_evals: number of evaluations of obj_func performed during line search
    """

    ls_evals = 0

    # First try
    t1 = increase * t
    f_1, g_1 = obj_func(x, t1, d)
    ls_evals += 1

    # Use quadratic interpolation to find the next step
    t_next = _quadratic_interpolate(0, f, gtd, t1, f_1, decrease, increase)
    f_next, g_next = obj_func(x, t_next, d)
    ls_evals += 1

    # If either t1 or t_next gives a decrease, accept the best one
    if min(f_1, f_next) < f:
        f_best, g_best, t_best = (
            (f_1, g_1, t1) if f_1 < f_next
            else (f_next, g_next, t_next)
        )
        return f_best, g_best, t_best, ls_evals

    # Otherwise backtrack
    t = t * decrease
    for _ in range(max_ls - 2):
        if t < min_step:
            print("sLBGS: Line search step size below minimum threshold, stopping line search.")
            return f, g_1, t, ls_evals
        f_new, g_new = obj_func(x, t, d)
        ls_evals += 1
        if f_new < f:
            return f_new, g_new, t, ls_evals
        t = t * decrease


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    
    # normalization factor to avoid arithmetic with large numbers
    s = torch.max(torch.max(d1.abs(),g1.abs()),g2.abs())
    d2_square = (d1/s)**2 - (g1/s) * (g2/s)
    if d2_square >= 0:
        d2 = s* d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # extrapolate 
        min_step = 1.1 * t  
        max_step = t * 4  
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # compute new trial value
        t = _cubic_interpolate(
            # pyrefly: ignore [index-error]
            # pyrefly: ignore [unbound-name]
            bracket[0],
            # pyrefly: ignore [unbound-name]
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            # pyrefly: ignore [index-error]
            # pyrefly: ignore [unbound-name]
            bracket[1],
            # pyrefly: ignore [unbound-name]
            bracket_f[1],
            # pyrefly: ignore [unbound-name]
            bracket_gtd[1],
        )

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        # pyrefly: ignore [unbound-name]
        eps = 0.1 * (max(bracket) - min(bracket))
        # pyrefly: ignore [unbound-name]
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            # pyrefly: ignore [unbound-name]
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                # pyrefly: ignore [unbound-name]
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    # pyrefly: ignore [unbound-name]
                    t = max(bracket) - eps
                else:
                    # pyrefly: ignore [unbound-name]
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        # pyrefly: ignore [unbound-name]
        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            # pyrefly: ignore [unsupported-operation]
            # pyrefly: ignore [unbound-name]
            bracket[high_pos] = t
            # pyrefly: ignore [unbound-name]
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            # pyrefly: ignore [unbound-name]
            bracket_gtd[high_pos] = gtd_new
            # pyrefly: ignore [unbound-name]
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            # pyrefly: ignore [index-error]
            # pyrefly: ignore [unbound-name]
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                # pyrefly: ignore [unsupported-operation]
                # pyrefly: ignore [unbound-name]
                bracket[high_pos] = bracket[low_pos]
                # pyrefly: ignore [unbound-name]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                # pyrefly: ignore [unbound-name]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            # pyrefly: ignore [unsupported-operation]
            # pyrefly: ignore [unbound-name]
            bracket[low_pos] = t
            # pyrefly: ignore [unbound-name]
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            # pyrefly: ignore [unbound-name]
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    # pyrefly: ignore [unbound-name]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return f_new, g_new, t, ls_func_evals


class sLBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float, optional): learning rate (default: 1)
        max_iter (int, optional): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int, optional): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float, optional): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float, optional): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int, optional): update history size (default: 100).
        line_search_fn (str, optional): either 'strong_wolfe' or None (default: None).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1,
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = {
            "lr": lr,
            "max_iter": max_iter,
            "max_eval": max_eval,
            "tolerance_grad": tolerance_grad,
            "tolerance_change": tolerance_change,
            "history_size": history_size,
            "line_search_fn": line_search_fn,
        }
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options (parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            # pyrefly: ignore [bad-assignment]
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update) -> None:
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        if offset != self._numel():
            raise AssertionError(f"Expected offset {offset} to equal {self._numel()}")

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data) -> None:
        for p, pdata in zip(self._params, params_data, strict=True):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d, backward=True):
        """backward keyword allows to avoid derivative computation during linesearch"""
        self._add_grad(t, d)
        loss = float(closure(backward=backward))
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.param_groups) != 1:
            raise AssertionError(
                f"Expected exactly one param_group, but got {len(self.param_groups)}"
            )

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = _to_scalar(group["lr"])
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        line_search_fn = group["line_search_fn"]
        history_size = group["history_size"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get("d")
        t = state.get("t")
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
        H_diag = state.get("H_diag")
        prev_flat_grad = state.get("prev_flat_grad")
        prev_loss = state.get("prev_loss")

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state["n_iter"] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state["n_iter"] == 1:
                t = min(1.0, 1.0 / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn == "quadratic_approx":
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d, backward=False)

                    loss, flat_grad, t, ls_func_evals = _quadratic_linesearch(
                        obj_func,
                        x_init,
                        t,
                        d,
                        loss,
                        gtd,
                        max_ls=max_eval - current_evals,
                    )

                elif line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func,
                        x_init,
                        t,
                        d,
                        loss,
                        flat_grad,
                        gtd,
                        max_ls=max_eval - current_evals,
                    )
                else:
                    raise RuntimeError("only 'strong_wolfe' or 'quadratic_approx' is supported")
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = closure()
                    loss = float(loss)
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["ro"] = ro
        state["H_diag"] = H_diag
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss
