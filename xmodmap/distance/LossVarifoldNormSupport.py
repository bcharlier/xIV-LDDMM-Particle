# import torch

# from xmodmap.distance.LossVarifoldNorm import LossVarifoldNorm
# from xmodmap.preprocess.preprocess import resizeData


# class LossVarifoldNormSupport(LossVarifoldNorm):
#     """
#     Varifold distance with support masks on either or both measures.

#         mode = "none"          plain varifold
#                "target"        source particles weighted by the TARGET mask (target observed region), evaluated at
#                                the deformed source positions qx1.                   [= the old Boundary loss]
#                "source"        target particles weighted by the SOURCE mask (source valid region), evaluated at the
#                                target particles pulled back to source space (Td, from backward shooting).
#                "intersection"  both, i.e. each measure is restricted to where the other one exists.

#     target_mask : mask defined in target space (needed for "target"/"intersection")
#     source_mask : mask defined in source space (needed for "source"/"intersection")
#     s, m        : unit-box rescaling, so masks can be queried in physical coordinates (resizeData(x, s, m))

#     Bandwidth convention (unchanged from LossVarifoldNormBoundary): __call__ receives the OPTIMISED variables
#     `lamb` / `lambS` (square roots) and squares them; normalize_across_scale and supportWeight* take the
#     EFFECTIVE bandwidth.

#     Source masking makes the target self-term (cst) depend on the deformation, so it is recomputed every call
#     (one extra M x M KeOps reduction per iteration, plus a backward shooting done by the model).
#     """

#     MODES = ("none", "target", "source", "intersection")

#     def __init__(self, sigmaVar, w_T, zeta_T, Ttilde, s=1., m=0.,
#                  target_mask=None, source_mask=None, mode="target",
#                  feature_kernel="euclidean", sigmaVarFeat=None, bandWidth=10):
#         if mode not in self.MODES:
#             raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
#         self.mode = mode
#         self.use_target_mask = mode in ("target", "intersection")
#         self.use_source_mask = mode in ("source", "intersection")
#         if self.use_target_mask and target_mask is None:
#             raise ValueError(f"mode={mode!r} needs a target_mask")
#         if self.use_source_mask and source_mask is None:
#             raise ValueError(f"mode={mode!r} needs a source_mask")
#         self.target_mask = target_mask
#         self.source_mask = source_mask
#         self.s = s
#         self.m = m

#         if w_T.dim() == 1:
#             w_T = w_T[..., None]
#         super().__init__(sigmaVar, w_T, zeta_T, Ttilde, feature_kernel=feature_kernel,
#                          sigmaVarFeat=sigmaVarFeat, bandWidth=bandWidth)
#         self.w_T0 = self.w_T          # unmasked target weights (self.w_T is swapped temporarily in normalisation)

#     def get_params(self):
#         params = super().get_params()
#         params["mode"] = self.mode
#         return params

#     # ------------------------------------------------------------------ masks
#     def supportWeightT(self, qx, lamb):
#         """Target-space mask at points qx given in the unit box. lamb = effective bandwidth. -> (N, 1)"""
#         return self.target_mask(resizeData(qx, self.s, self.m), lamb).reshape(-1, 1)

#     def supportWeightS(self, Td, lambS):
#         """Source-space mask at pulled-back target points Td given in the unit box. -> (M, 1)"""
#         return self.source_mask(resizeData(Td, self.s, self.m), lambS).reshape(-1, 1)

#     # name used by the save helpers in the run scripts
#     supportWeight = supportWeightT

#     # ------------------------------------------------------------------ normalisation (beta)
#     def normalize_across_scale(self, Stilde, w_S, zeta_S, pi_STinit, lamb0=None, lambS0=None):
#         """
#         Calibrate beta on the masked measures at the identity state (qx = Stilde, Td = Ttilde), so the initial
#         data term is again sum_sigma 0.6 / sigma regardless of mode. lamb0 / lambS0: effective bandwidths.
#         """
#         wS = w_S * self.supportWeightT(Stilde, lamb0) if self.use_target_mask else w_S
#         wT = self.w_T0 * self.supportWeightS(self.Ttilde, lambS0) if self.use_source_mask else self.w_T0
#         self.w_T = wT
#         try:
#             self.set_normalization(Stilde, wS, zeta_S, pi_STinit)
#         finally:
#             self.w_T = self.w_T0
#             self.update_cst_and_K()   # restore cst computed with the unmasked target weights

#     # ------------------------------------------------------------------ data term
#     def __call__(self, sSx, sSw, zeta_S, pi_ST, lamb=None, Td=None, lambS=None):
#         """
#         sSx, sSw : deformed source positions / weights (qx1, qw1)
#         Td       : target particles pulled back to source space (required when the source mask is active)
#         lamb, lambS : optimised (sqrt-parametrised) bandwidth variables for the target / source mask
#         """
#         if self.use_target_mask:
#             sSw = sSw * self.supportWeightT(sSx, lamb ** 2)

#         if not self.use_source_mask:
#             return super().__call__(sSx, sSw, zeta_S, pi_ST)

#         if Td is None:
#             raise ValueError(f"mode={self.mode!r} needs Td (target pulled back to source space)")
#         wT = self.w_T0 * self.supportWeightS(Td, lambS ** 2)

#         nu_Spi = (sSw * zeta_S) @ (pi_ST ** 2)
#         cst = 0.0
#         k1 = 0.0
#         k2 = 0.0
#         for i, (b0, b1) in enumerate(self.bands):
#             nu_band = nu_Spi[:, b0:b1].contiguous()
#             wz_band = (wT * self.zeta_T[:, b0:b1]).contiguous()
#             cst = cst + self.K[i](self.Ttilde, self.Ttilde, wz_band, wz_band).sum()
#             k1 = k1 + self.K[i](sSx, sSx, nu_band, nu_band).sum()
#             k2 = k2 + self.K[i](sSx, self.Ttilde, nu_band, wz_band).sum()
#         return (1.0 / 2.0) * (cst + k1 - 2.0 * k2)



# Updated code 11/07

import torch

from xmodmap.distance.LossVarifoldNorm import LossVarifoldNorm
from xmodmap.preprocess.preprocess import resizeData


class LossVarifoldNormSupport(LossVarifoldNorm):
    """
    Varifold distance with support masks on either or both measures.

        mode = "none"          plain varifold
               "target"        source particles weighted by the TARGET mask, evaluated at
                               the deformed source positions qx1.                   [= the old Boundary loss]
               "source"        target particles weighted by the SOURCE mask, evaluated at the
                               target particles pulled back to source space (Td, from backward shooting).
               "intersection"  both, i.e. each measure is restricted to where the other one exists.

    target_mask : mask defined in target space
    source_mask : mask defined in source space
    s, m        : unit-box rescaling, so masks can be queried in physical coordinates (resizeData(x, s, m))

    Bandwidth convention (unchanged from LossVarifoldNormBoundary): __call__ receives the OPTIMISED variables
    `lamb` / `lambS` (square roots) and squares them; normalize_across_scale and supportWeight* take the
    EFFECTIVE bandwidth.
    """

    MODES = ("none", "target", "source", "intersection")

    def __init__(self, sigmaVar, w_T, zeta_T, Ttilde, s=1., m=0.,
                 target_mask=None, source_mask=None, mode="target",
                 feature_kernel="euclidean", sigmaVarFeat=None, bandWidth=10):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.use_target_mask = mode in ("target", "intersection")
        self.use_source_mask = mode in ("source", "intersection")
        if self.use_target_mask and target_mask is None:
            raise ValueError(f"mode={mode!r} needs a target_mask")
        if self.use_source_mask and source_mask is None:
            raise ValueError(f"mode={mode!r} needs a source_mask")
        self.target_mask = target_mask
        self.source_mask = source_mask
        self.s = s
        self.m = m

        if w_T.dim() == 1:
            w_T = w_T[..., None]
        super().__init__(sigmaVar, w_T, zeta_T, Ttilde, feature_kernel=feature_kernel,
                         sigmaVarFeat=sigmaVarFeat, bandWidth=bandWidth)
        self.w_T0 = self.w_T      
        # Mask coverage:  c_S = fraction of the deformed source mass inside the target mask,
        #                 c_T = fraction of the target mass whose pullback lies inside the source mask.
        self.c_S = self.c_T = None
        self.c_S0 = self.c_T0 = None

    def get_params(self):
        params = super().get_params()
        params["mode"] = self.mode
        return params

    # ------------------------------------------------------------------ masks
    def supportWeightT(self, qx, lamb):
        """Target-space mask at points qx given in the unit box. lamb = effective bandwidth. -> (N, 1)"""
        return self.target_mask(resizeData(qx, self.s, self.m), lamb).reshape(-1, 1)

    def supportWeightS(self, Td, lambS):
        """Source-space mask at pulled-back target points Td given in the unit box. -> (M, 1)"""
        return self.source_mask(resizeData(Td, self.s, self.m), lambS).reshape(-1, 1)

    # name used by the save helpers in the run scripts
    supportWeight = supportWeightT

    # ------------------------------------------------------------------ normalisation (beta)
    def normalize_across_scale(self, Stilde, w_S, zeta_S, pi_STinit, lamb0=None, lambS0=None):
        """
        Calibrate beta on the masked measures at the identity state (qx = Stilde, Td = Ttilde), so the initial
        data term is again sum_sigma 0.6 / sigma regardless of mode. lamb0 / lambS0: effective bandwidths.
        """
        wS = w_S * self.supportWeightT(Stilde, lamb0) if self.use_target_mask else w_S
        wT = self.w_T0 * self.supportWeightS(self.Ttilde, lambS0) if self.use_source_mask else self.w_T0
        self.c_S0 = float(wS.sum() / w_S.sum()) if self.use_target_mask else None      # initial coverage
        self.c_T0 = float(wT.sum() / self.w_T0.sum()) if self.use_source_mask else None
        self.w_T = wT
        try:
            self.set_normalization(Stilde, wS, zeta_S, pi_STinit)
        finally:
            self.w_T = self.w_T0
            self.update_cst_and_K()   # restore cst computed with the unmasked target weights

    #  data term
    def __call__(self, sSx, sSw, zeta_S, pi_ST, lamb=None, Td=None, lambS=None):
        """
        sSx, sSw : deformed source positions / weights (qx1, qw1)
        Td       : target particles pulled back to source space (required when the source mask is active)
        lamb, lambS : optimised (sqrt-parametrised) bandwidth variables for the target / source mask
        """
        if self.use_target_mask:
            a = self.supportWeightT(sSx, lamb ** 2)
            self.c_S = (sSw * a).sum() / sSw.sum()
            sSw = sSw * a

        if not self.use_source_mask:
            return super().__call__(sSx, sSw, zeta_S, pi_ST)

        if Td is None:
            raise ValueError(f"mode={self.mode!r} needs Td (target pulled back to source space)")
        b = self.supportWeightS(Td, lambS ** 2)
        wT = self.w_T0 * b
        self.c_T = wT.sum() / self.w_T0.sum()

        nu_Spi = (sSw * zeta_S) @ (pi_ST ** 2)
        cst = 0.0
        k1 = 0.0
        k2 = 0.0
        for i, (b0, b1) in enumerate(self.bands):
            nu_band = nu_Spi[:, b0:b1].contiguous()
            wz_band = (wT * self.zeta_T[:, b0:b1]).contiguous()
            cst = cst + self.K[i](self.Ttilde, self.Ttilde, wz_band, wz_band).sum()
            k1 = k1 + self.K[i](sSx, sSx, nu_band, nu_band).sum()
            k2 = k2 + self.K[i](sSx, self.Ttilde, nu_band, wz_band).sum()
        return (1.0 / 2.0) * (cst + k1 - 2.0 * k2)

