import inspect
import os
import time
import torch

from xmodmap.optimizer.config import lbfgsConfigDefaults
from xmodmap.optimizer.sLBFGS import sLBFGS

class Model:
    variables = None
    variables_to_optimize = None
    precond = lambda *x: x  # identity function
    optimizer = None
    log = []
    current_step = 0
    _savedir = None

    def __init__(self, *args):
        self.logs = {"full": [], "steps": []}
        self.log = self.logs["full"]
        pass

    def get_params(self):
        pass

    def get_variables_optimized(self):
        # set gradients to zero before returning
        self.optimizer.zero_grad()
        return self.precond(self.variables)

    def init(self, variables, variables_to_optimize, precond=None, savedir=os.path.join(os.getcwd(), "output")):
        # a dict containing all the variables of the model
        self.variables = variables

        # a list of the variables to optimize
        self.variables_to_optimize = {key: self.variables[key] for i, key in enumerate(variables_to_optimize)}

        # no precond by default, see self.set_precond()
        if precond is not None:
            self.set_precond(precond)

        # init the optimizer
        self.set_optimizer()

        # init the logger
        self.savedir = savedir

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        os.makedirs(savedir, exist_ok=True)
        self._savedir = savedir

    def set_optimizer(self, state=None):
        self.optimizer = sLBFGS(self.variables_to_optimize.values(), **lbfgsConfigDefaults)
        #self.optimizer = torch.optim.Adam(self.variables_to_optimize.values(), lr=1e-7)
        if state is not None:
            self.optimizer.load_state_dict(state)

    def set_precond(self, preprecond):
        if isinstance(preprecond, dict):  # if preprecond is a dict, should store weights
            # check if the keys of preprecond are in self.variables
            assert preprecond.keys() <= self.variables.keys()
            # init a dict with value 1
            precond = {key: 1.0 for key in self.variables}
            # update the dict with the kwargs
            precond.update(preprecond)

            # create a function that returns the precond for each variable
            self.precond = lambda x: {key: x[key] * precond[key] for key in self.variables}
        elif callable(preprecond):  # if prepreconf is a function
            # check if the keys output by preprecond are compatible with self.variables
            assert preprecond(self.variables).keys() == inspect.signature(self.loss).parameters.keys()
            self.precond = preprecond

        self.precondWeights = preprecond

    def loss(self, **kwargs):
        pass

    def closure(self, backward=True):
        self.optimizer.zero_grad()

        losses = self.loss(**self.precond(self.variables))
        loss = sum(losses)
        self.logs["full"].append([l.detach().cpu() for l in losses])

        # Prevent to return a nan when too large values are
        if torch.isnan(loss):
            loss = torch.inf
            for v in self.variables_to_optimize.values():
                v.grad = torch.full_like(v, float("inf"))
        elif backward:
            loss.backward()

        return loss

    def optimize(self, steps=100, wall_time=None, cb=None):
        self.steps = steps

        print("performing optimization...")
        self.start_optim = time.time()

        for i in range(self.current_step, self.current_step + self.steps):

            print("it ", i, ": ", end="")

            # Perform a step of optimization, and save the current state
            loss_step = self.optimizer.step(self.closure)
            self.logs["steps"].append(loss_step.detach().cpu())

            self.saveState()

            if cb is not None:
                cb(self)

            if self.stoppingCondition(wall_time=wall_time):
                break

            self.current_step += 1

        print("Optimization time: ", round(time.time() - self.start_optim, 2), " seconds")

    def stoppingCondition(self, wall_time=None):

        # check for NaN
        if torch.isnan(self.logs["steps"][-1]):
            print(f"Stopping: NaN encountered at iteration {self.current_step}.")
            return 1

        # check for convergence
        if (self.current_step > 0) and torch.allclose(self.logs["steps"][-1], self.logs["steps"][-2]):
            print(f"Stopping: Local minimum reached at iteration {self.current_step}.")
            return 1

        if self.current_step == self.steps:
            print(f"Stopping: Maximum number {self.steps} of iterations reached.")
            return 1

        # Check time limit
        if wall_time is not None and (time.time() - self.start_optim) >= wall_time:
            print(f"Stopping: time limit {wall_time}s reached.")
            return 1

        return 0

    def saveState(self):
        """
        osd = state of optimizer
        its = total iterations
        i = current iteration
        xopt = current optimization variable (p0*pTilde)
        """

        checkpoint = {
            "variables_to_optimize": self.variables_to_optimize,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "precondWeights": self.precondWeights,
            "steps": self.steps,
            "current_step": self.current_step,
            "log": self.log,
            "savedir": self.savedir,
        }
        checkpoint.update(self.get_params())

        filename = os.path.join(self.savedir, f"checkpoint.pt")
        torch.save(checkpoint, filename)

    def resume(self, variables, filename):
        print(f"Resuming optimization from {filename}. Loading... ", end="")

        checkpoint = torch.load(filename)

        self.variables = variables
        self.variables.update(checkpoint["variables_to_optimize"])
        self.variables_to_optimize = checkpoint["variables_to_optimize"]

        self.set_optimizer(state=checkpoint["optimizer_state_dict"])
        self.set_precond(checkpoint["precondWeights"])

        self.steps = checkpoint["steps"]
        self.current_step = checkpoint["current_step"]
        self.log = checkpoint["log"]
        self.savedir = checkpoint["savedir"]

        self.check_resume(checkpoint)

        print("done.")

    def check_resume(self, checkpoint):
        pass

    def print_log(self, logScale=False):
        pass
