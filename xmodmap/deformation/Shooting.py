class Abstract_Shooting:
    def __init__(self, hamiltonian, nt=10):
        self.hamiltonian = hamiltonian
        self.nt = nt

    @staticmethod
    def ralstonIntegrator(ODESystem, x0, nt, deltat=1.0):
        x = tuple(v.clone() for v in x0)
        dt = deltat / nt
        trajectory = [x]
        for _ in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(v + (2 * dt / 3) * vdot for v, vdot in zip(x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(
                v + 0.25 * dt * (vdot + 3 * vdoti)
                for v, vdot, vdoti in zip(x, xdot, xdoti)
            )
            trajectory.append(x)
        return trajectory


class Shooting(Abstract_Shooting):
    def __call__(self, px0, pw0, qx0, qw0):
        return self.ralstonIntegrator(
            self.hamiltonian.forward_system, (px0, pw0, qx0, qw0), self.nt
        )


class ShootingGrid(Abstract_Shooting):
    def __call__(self, px0, pw0, qx0, qw0, qGrid, qGridw):
        return self.ralstonIntegrator(
            self.hamiltonian.system_with_points,
            (px0, pw0, qx0, qw0, qGrid, qGridw),
            self.nt,
        )


class ShootingBackwards(Abstract_Shooting):
    def __call__(self, px1, pw1, qx1, qw1, T, wT):
        return self.ralstonIntegrator(
            self.hamiltonian.system_with_points,
            (-px1, -pw1, qx1, qw1, T, wT),
            self.nt,
        )
