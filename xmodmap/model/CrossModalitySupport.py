# import torch

# from xmodmap.model.CrossModalityBoundary import CrossModalityBoundary


# class CrossModalitySupport(CrossModalityBoundary):
#     """
#     CrossModalityBoundary with optional source-side masking.

#     Extra inputs vs. CrossModalityBoundary:
#         shootingBack : xmodmap.deformation.ShootingBackwards, used to pull Ttilde back to source space (Td)
#                        every iteration when dataLoss.use_source_mask is True (may be None otherwise)
#         Ttilde, w_T  : target positions (unit box) and weights that get pulled back

#     Optimised variables passed to loss(): `lamb` (target-mask bandwidth) and `lambS` (source-mask bandwidth);
#     each is only used, and only regularised by lambLoss, when its mask is active. The loss still returns the
#     4-tuple (H, data, pi, lambda) so logging, CSV export and print_log are unchanged.
#     """

#     def __init__(self, hamiltonian, shooting, shootingBack, dataLoss, piLoss, lambLoss, Ttilde, w_T):
#         super().__init__(hamiltonian, shooting, dataLoss, piLoss, lambLoss)
#         self.shootingBack = shootingBack
#         self.Ttilde = Ttilde
#         self.w_T = w_T
#         if dataLoss.use_source_mask and shootingBack is None:
#             raise ValueError(f"mode={dataLoss.mode!r} needs shootingBack")

#     def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None, lamb=None, lambS=None):
#         hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

#         px1, pw1, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]

#         Td = None
#         if self.dataLoss.use_source_mask:
#             _, _, _, _, Td, _ = self.shootingBack(px1, pw1, qx1, qw1, self.Ttilde, self.w_T)[-1]

#         dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, pi_ST, lamb, Td=Td, lambS=lambS)
#         pLoss = self.piLoss.weight * self.piLoss(qw1, pi_ST)

#         lLoss = torch.zeros((), dtype=dLoss.dtype, device=dLoss.device)
#         if self.dataLoss.use_target_mask:
#             lLoss = lLoss + self.lambLoss.weight * self.lambLoss(lamb)
#         if self.dataLoss.use_source_mask:
#             lLoss = lLoss + self.lambLoss.weight * self.lambLoss(lambS)

#         print(f"loss: {(hLoss + dLoss + pLoss + lLoss).detach().cpu().numpy()}; "
#               f"H loss: {hLoss.detach().cpu().numpy()}; "
#               f"Var loss: {dLoss.detach().cpu().numpy()}; "
#               f"Pi loss: {pLoss.detach().cpu().numpy()}; "
#               f"Lambda loss: {lLoss.detach().cpu().numpy()}")

#         return hLoss, dLoss, pLoss, lLoss


# # Updated Code 11/07

# import torch

# from xmodmap.model.CrossModalityBoundary import CrossModalityBoundary


# class CrossModalitySupport(CrossModalityBoundary):
#     """
#     CrossModalityBoundary with optional source-side masking and safeguards against degenerate solutions.

#     Extra inputs vs. CrossModalityBoundary:
#         shootingBack : xmodmap.deformation.ShootingBackwards, used to pull Ttilde back to source space (Td)
#                        every iteration when dataLoss.use_source_mask is True (may be None otherwise)
#         Ttilde, w_T  : target positions (unit box) and weights that get pulled back

#     Safeguards (all optional, all added to the 4th loss slot so logging, CSV export and print_log are unchanged):
#         areaReg, rho_area, area_tol, local_tol :
#             area budget. areaReg is an xmodmap.distance.AreaRegularizer.LocalAreaChange built on the source particles.
#             g = mean log det F of the deformation; penalty rho_area * hinge(g) is zero while -tol_shrink <= g <= tol_expand
#             (area_tol: a number or a (tol_shrink, tol_expand) pair). Stops the source from collapsing or blowing up.
#         rho_cov :
#             mask-weight regulariser. The mean mask weight, i.e. the fraction of source mass inside the target mask (c_S)
#             and of target mass whose pullback is inside the source mask (c_T), is penalised when it falls below its
#             initial value (identity deformation):
#             rho_cov * (relu(c_S0 - c_S)^2 + relu(c_T0 - c_T)^2). Stops the masks from emptying.

#     Optimised variables passed to loss(): `lamb` (target-mask bandwidth) and `lambS` (source-mask bandwidth);
#     each is only used, and only regularised by lambLoss, when its mask is active.
#     """

#     def __init__(self, hamiltonian, shooting, shootingBack, dataLoss, piLoss, lambLoss, Ttilde, w_T,
#                  areaReg=None, area_tol=0.3, local_tol=None, rho_area=0.0, rho_cov=0.0):
#         super().__init__(hamiltonian, shooting, dataLoss, piLoss, lambLoss)
#         self.shootingBack = shootingBack
#         self.Ttilde = Ttilde
#         self.w_T = w_T
#         self.areaReg = areaReg
#         self.area_tol = area_tol
#         self.local_tol = local_tol
#         self.rho_area = rho_area
#         self.rho_cov = rho_cov
#         if dataLoss.use_source_mask and shootingBack is None:
#             raise ValueError(f"mode={dataLoss.mode!r} needs shootingBack")

#     def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None, lamb=None, lambS=None):
#         hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

#         px1, pw1, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]

#         Td = None
#         if self.dataLoss.use_source_mask:
#             _, _, _, _, Td, _ = self.shootingBack(px1, pw1, qx1, qw1, self.Ttilde, self.w_T)[-1]

#         dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, pi_ST, lamb, Td=Td, lambS=lambS)
#         pLoss = self.piLoss.weight * self.piLoss(qw1, pi_ST)

#         zero = torch.zeros((), dtype=dLoss.dtype, device=dLoss.device)

#         # bandwidth priors
#         lambPen = zero
#         if self.dataLoss.use_target_mask:
#             lambPen = lambPen + self.lambLoss.weight * self.lambLoss(lamb)
#         if self.dataLoss.use_source_mask:
#             lambPen = lambPen + self.lambLoss.weight * self.lambLoss(lambS)

#         # coverage floors
#         covPen = zero
#         covText = ""
#         for name, c, c0 in (("S", self.dataLoss.c_S, self.dataLoss.c_S0), ("T", self.dataLoss.c_T, self.dataLoss.c_T0)):
#             if c is None or c0 is None:
#                 continue
#             covText += f"; cover {name}: {c.detach().cpu().numpy():.3f} (init {c0:.3f})"
#             if self.rho_cov > 0:
#                 covPen = covPen + self.rho_cov * torch.relu(c0 - c) ** 2

#         # area budget
#         areaPen = zero
#         areaText = ""
#         if self.areaReg is not None:
#             pen, g = self.areaReg.penalty(qx1, self.area_tol, self.local_tol)
#             areaText = f"; area ratio: {torch.exp(g).detach().cpu().numpy():.3f}"
#             if self.rho_area > 0:
#                 areaPen = self.rho_area * pen

#         lLoss = lambPen + covPen + areaPen

#         print(f"loss: {(hLoss + dLoss + pLoss + lLoss).detach().cpu().numpy()}; "
#               f"H loss: {hLoss.detach().cpu().numpy()}; "
#               f"Var loss: {dLoss.detach().cpu().numpy()}; "
#               f"Pi loss: {pLoss.detach().cpu().numpy()}; "
#               f"Reg loss: {lLoss.detach().cpu().numpy()} "
#               f"(lambda {lambPen.detach().cpu().numpy():.4f}, cover {covPen.detach().cpu().numpy():.4f}, "
#               f"area {areaPen.detach().cpu().numpy():.4f})"
#               f"{covText}{areaText}")

#         return hLoss, dLoss, pLoss, lLoss


# Updated code 11/08

import torch

from xmodmap.model.CrossModalityBoundary import CrossModalityBoundary


class CrossModalitySupport(CrossModalityBoundary):
    """
    CrossModalityBoundary with optional source-side masking and safeguards against degenerate solutions.

    Extra inputs vs CrossModalityBoundary:
        shootingBack : xmodmap.deformation.ShootingBackwards, used to pull Ttilde back to source space (Td)
                       every iteration when dataLoss.use_source_mask is True (may be None otherwise)
        Ttilde, w_T  : target positions (unit box) and weights that get pulled back

        rho_cov :
            mask-weight regulariser. c_S = sum_i a_i w_i / sum_i w_i is the mean target-mask weight of the deformed source
            particles, c_T = sum_j b_j v_j / sum_j v_j the mean source-mask weight of the pulled-back target points:
            rho_cov * [(1 - c_S)^2 + (1 - c_T)^2], with only the terms of the active masks:
            mode "target" -> c_S term, "source" -> c_T term, "intersection" -> both

    Optimised variables passed to loss(): `lamb` (target-mask bandwidth) and `lambS` (source-mask bandwidth);
    each is only used, and only regularised by lambLoss, when its mask is active.
    """

    def __init__(self, hamiltonian, shooting, shootingBack, dataLoss, piLoss, lambLoss, Ttilde, w_T,
                 areaReg=None, area_tol=0.3, local_tol=None, rho_area=0.0, rho_cov=0.0):
        super().__init__(hamiltonian, shooting, dataLoss, piLoss, lambLoss)
        self.shootingBack = shootingBack
        self.Ttilde = Ttilde
        self.w_T = w_T
        self.areaReg = areaReg
        self.area_tol = area_tol
        self.local_tol = local_tol
        self.rho_area = rho_area
        self.rho_cov = rho_cov
        if dataLoss.use_source_mask and shootingBack is None:
            raise ValueError(f"mode={dataLoss.mode!r} needs shootingBack")

    def loss(self, px=None, pw=None, qx=None, qw=None, pi_ST=None, zeta_S=None, lamb=None, lambS=None):
        hLoss = self.hamiltonian.weight * self.hamiltonian(px, pw, qx, qw)

        px1, pw1, qx1, qw1 = self.shooting(px, pw, qx, qw)[-1]

        Td = None
        if self.dataLoss.use_source_mask:
            _, _, _, _, Td, _ = self.shootingBack(px1, pw1, qx1, qw1, self.Ttilde, self.w_T)[-1]

        dLoss = self.dataLoss.weight * self.dataLoss(qx1, qw1, zeta_S, pi_ST, lamb, Td=Td, lambS=lambS)
        pLoss = self.piLoss.weight * self.piLoss(qw1, pi_ST)

        zero = torch.zeros((), dtype=dLoss.dtype, device=dLoss.device)

        # bandwidth priors <<-- from the partialNN module
        lambPen = zero
        if self.dataLoss.use_target_mask:
            lambPen = lambPen + self.lambLoss.weight * self.lambLoss(lamb)
        if self.dataLoss.use_source_mask:
            lambPen = lambPen + self.lambLoss.weight * self.lambLoss(lambS)

        # coverage floors
        covPen = zero
        covText = ""
        for name, c, c0 in (("S", self.dataLoss.c_S, self.dataLoss.c_S0), ("T", self.dataLoss.c_T, self.dataLoss.c_T0)):
            if c is None or c0 is None:
                continue
            covText += f"; cover {name}: {c.detach().cpu().numpy():.3f} (init {c0:.3f})"
            if self.rho_cov > 0:
                covPen = covPen + self.rho_cov * (1.0 - c) ** 2

        # area budget (we are not using this)
        areaPen = zero
        areaText = ""
        if self.areaReg is not None:
            pen, g = self.areaReg.penalty(qx1, self.area_tol, self.local_tol)
            areaText = f"; area ratio: {torch.exp(g).detach().cpu().numpy():.3f}"
            if self.rho_area > 0:
                areaPen = self.rho_area * pen

        lLoss = lambPen + covPen + areaPen

        print(f"loss: {(hLoss + dLoss + pLoss + lLoss).detach().cpu().numpy()}; "
              f"H loss: {hLoss.detach().cpu().numpy()}; "
              f"Var loss: {dLoss.detach().cpu().numpy()}; "
              f"Pi loss: {pLoss.detach().cpu().numpy()}; "
              f"Reg loss: {lLoss.detach().cpu().numpy()} "
              f"(lambda {lambPen.detach().cpu().numpy():.4f}, cover {covPen.detach().cpu().numpy():.4f}, "
              f"area {areaPen.detach().cpu().numpy():.4f})"
              f"{covText}{areaText}")

        return hLoss, dLoss, pLoss, lLoss