# """
# Support masks: soft "is this point inside the observed region?" indicators for the varifold data term.

# Every mask is a callable

#         mask(x, lamb) -> (N, 1) tensor in [0, 1]

# where `x` is (N, d) in the data's *physical* coordinates (the loss converts from the unit box with
# resizeData before calling) and `lamb` is the *effective* transition width (already squared, see
# LossVarifoldNormSupport). Works for d = 2 (images) and d = 3 (volumes).

# SDFSupport is the default: weight = 0.5 * (1 + tanh(sdf(x) / lamb)), with sdf the signed distance to the
# region boundary (positive inside). lamb is therefore a width in the same physical units as the data.
# NNetSupport wraps the existing BoundaryClassifier so the old gene-data runs keep working.
# """
# import numpy as np
# import torch
# import torch.nn.functional as F
# from scipy import ndimage as ndi
# from scipy.spatial import cKDTree


# class SupportMask:
#     def __call__(self, x, lamb):
#         raise NotImplementedError


# class SDFSupport(SupportMask):
#     """
#     Signed-distance-function mask on a regular grid, sampled with (bi/tri)linear interpolation, so it is
#     differentiable w.r.t. the query points.

#     sdf     : array (n_0, ..., n_{d-1}) indexed [x, y(, z)], positive inside, in physical length units
#     spacing : voxel size (scalar or length-d)
#     origin  : physical coordinate of the CENTRE of voxel [0, 0(, 0)]
#     """

#     def __init__(self, sdf, spacing=1.0, origin=0.0):
#         sdf = np.asarray(sdf, dtype=np.float32)
#         self.d = sdf.ndim
#         if self.d not in (2, 3):
#             raise ValueError(f"SDFSupport supports 2D or 3D grids, got {self.d}D")
#         self.spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float32), (self.d,)).copy()
#         self.origin = np.broadcast_to(np.asarray(origin, dtype=np.float32), (self.d,)).copy()
#         self.shape = sdf.shape
#         self._sdf = sdf
#         self._cache = {}

#     # ------------------------------------------------------------------ constructors
#     @staticmethod
#     def _signed_distance(inside, spacing):
#         """edt(inside) - edt(outside): +h at the first inside voxel, -h at the first outside one, so linear
#         interpolation puts the zero level set exactly halfway between them (the true boundary)."""
#         if not inside.any():
#             raise ValueError("support mask is empty")
#         if inside.all():
#             raise ValueError("support mask has no background; pad it before building the SDF")
#         return ndi.distance_transform_edt(inside, sampling=spacing) - ndi.distance_transform_edt(~inside, sampling=spacing)

#     @classmethod
#     def from_mask(cls, mask, spacing=1.0, origin=0.0, pad=10):
#         """mask: bool array [x, y(, z)] (True = observed). Padded by `pad` background voxels on every side so the
#         sdf is defined (and negative) outside the field of view; queries beyond the grid clamp to the border."""
#         mask = np.asarray(mask).astype(bool)
#         d = mask.ndim
#         sp = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (d,))
#         org = np.broadcast_to(np.asarray(origin, dtype=np.float64), (d,))
#         padded = np.pad(mask, pad, mode="constant", constant_values=False)
#         return cls(cls._signed_distance(padded, sp), sp, org - pad * sp)

#     @classmethod
#     def from_particles(cls, points, res=None, closing=2, fill_holes=True, pad=10, keep=None):
#         """
#         Mask from a particle cloud: the region where particles exist is 'observed'.

#         points : (N, d) array/tensor in physical coordinates
#         res    : grid cell size. None -> 1.5 x median nearest-neighbour distance (robust to ds_factor
#                  downsampling, which makes the particle spacing coarser)
#         closing: binary-closing iterations to bridge gaps between particles
#         fill_holes: fill interior holes (set False if holes, e.g. a lesion, are real and should be masked)
#         keep   : optional bool array selecting which particles count as 'observed'
#         """
#         pts = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
#         pts = pts.astype(np.float64)
#         if keep is not None:
#             pts = pts[np.asarray(keep).astype(bool)]
#         d = pts.shape[1]
#         if res is None:
#             sub = pts if len(pts) <= 50000 else pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)]
#             res = 1.5 * float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
#         lo = pts.min(axis=0)
#         idx = np.floor((pts - lo) / res).astype(int)
#         occ = np.zeros(idx.max(axis=0) + 1, dtype=bool)
#         occ[tuple(idx.T)] = True
#         occ = np.pad(occ, pad, mode="constant", constant_values=False)   # pad BEFORE closing so edges don't erode
#         if closing > 0:
#             occ = ndi.binary_closing(occ, structure=ndi.generate_binary_structure(d, 1), iterations=closing)
#         if fill_holes:
#             occ = ndi.binary_fill_holes(occ)
#         origin = lo + 0.5 * res - pad * res
#         return cls(cls._signed_distance(occ, np.full(d, res)), np.full(d, res), origin)

#     # ------------------------------------------------------------------ evaluation
#     def _on(self, device):
#         key = str(device)
#         if key not in self._cache:
#             vol = torch.as_tensor(self._sdf).permute(*range(self.d - 1, -1, -1))[None, None].contiguous().to(device)
#             lo = torch.as_tensor(self.origin, device=device)
#             hi = lo + (torch.tensor(self.shape, device=device, dtype=lo.dtype) - 1) * torch.as_tensor(self.spacing, device=device)
#             self._cache[key] = (vol, lo, hi)
#         return self._cache[key]

#     def sdf_at(self, x):
#         """Signed distance at physical points x (N, d). Differentiable w.r.t. x."""
#         vol, lo, hi = self._on(x.device)
#         g = 2.0 * (x.to(vol.dtype) - lo) / (hi - lo) - 1.0                      # normalised to [-1, 1]
#         f = F.grid_sample(vol, g.reshape((1, -1) + (1,) * (self.d - 1) + (self.d,)),
#                           mode="bilinear", padding_mode="border", align_corners=True)
#         return f.reshape(-1).to(x.dtype)

#     def __call__(self, x, lamb):
#         return (0.5 * (1.0 + torch.tanh(self.sdf_at(x) / lamb))).reshape(-1, 1)


# class NNetSupport(SupportMask):
#     """Wraps the existing BoundaryClassifier (3D only): weight = 0.5 * (1 + tanh(f(x) / lamb))."""

#     def __init__(self, nnetFile):
#         import xmodmap
#         self.bc = xmodmap.preprocess.BoundaryClassifier(param_file=nnetFile)

#     def __call__(self, x, lamb):
#         return self.bc.predict(1.0 / lamb, x).reshape(-1, 1)


# Updated code 11/07

"""
Support masks: soft "is this point inside the observed region?" indicators for the varifold data term.

Every mask is a callable

        mask(x, lamb) -> (N, 1) tensor in [0, 1]

where `x` is (N, d) in the data's *physical* coordinates (the loss converts from the unit box with
resizeData before calling) and `lamb` is the *effective* transition width (already squared, see
LossVarifoldNormSupport). Works for d = 2 (images) and d = 3 (volumes).

SDFSupport is the default: weight = 0.5 * (1 + tanh(sdf(x) / lamb)), with sdf the signed distance to the
region boundary (positive inside). lamb is therefore a width in the same physical units as the data.
NNetSupport wraps the existing BoundaryClassifier so the old gene-data runs keep working.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


class SupportMask:
    def __call__(self, x, lamb):
        raise NotImplementedError


class SDFSupport(SupportMask):
    """
    Signed-distance-function mask on a regular grid, sampled with (bi/tri)linear interpolation, so it is
    differentiable w.r.t. the query points.

    sdf     : array (n_0, ..., n_{d-1}) indexed [x, y(, z)], positive inside, in physical length units
    spacing : voxel size (scalar or length-d)
    origin  : physical coordinate of the CENTRE of voxel [0, 0(, 0)]
    """

    def __init__(self, sdf, spacing=1.0, origin=0.0):
        sdf = np.asarray(sdf, dtype=np.float32)
        self.d = sdf.ndim
        if self.d not in (2, 3):
            raise ValueError(f"SDFSupport supports 2D or 3D grids, got {self.d}D")
        self.spacing = np.broadcast_to(np.asarray(spacing, dtype=np.float32), (self.d,)).copy()
        self.origin = np.broadcast_to(np.asarray(origin, dtype=np.float32), (self.d,)).copy()
        self.shape = sdf.shape
        self._sdf = sdf
        self._cache = {}

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def _signed_distance(inside, spacing):
        """edt(inside) - edt(outside): +h at the first inside voxel, -h at the first outside one, so linear
        interpolation puts the zero level set exactly halfway between them (the true boundary)."""
        if not inside.any():
            raise ValueError("support mask is empty")
        if inside.all():
            raise ValueError("support mask has no background; pad it before building the SDF")
        return ndi.distance_transform_edt(inside, sampling=spacing) - ndi.distance_transform_edt(~inside, sampling=spacing)

    @classmethod
    def from_mask(cls, mask, spacing=1.0, origin=0.0, pad=10):
        """mask: bool array [x, y(, z)] (True = observed). Padded by `pad` background voxels on every side so the
        sdf is defined (and negative) outside the field of view; queries beyond the grid clamp to the border."""
        mask = np.asarray(mask).astype(bool)
        d = mask.ndim
        sp = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (d,))
        org = np.broadcast_to(np.asarray(origin, dtype=np.float64), (d,))
        padded = np.pad(mask, pad, mode="constant", constant_values=False)
        return cls(cls._signed_distance(padded, sp), sp, org - pad * sp)

    @classmethod
    def from_particles(cls, points, res=None, closing=2, fill_holes=True, pad=10, keep=None):
        """
        Mask from a particle cloud: the region where particles exist is 'observed'.

        points : (N, d) array/tensor in physical coordinates
        res    : grid cell size. None -> 1.5 x median nearest-neighbour distance (robust to ds_factor
                 downsampling, which makes the particle spacing coarser)
        closing: binary-closing iterations to bridge gaps between particles
        fill_holes: fill interior holes (set False if holes, e.g. a lesion, are real and should be masked)
        keep   : optional bool array selecting which particles count as 'observed'
        """
        pts = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
        pts = pts.astype(np.float64)
        if keep is not None:
            pts = pts[np.asarray(keep).astype(bool)]
        d = pts.shape[1]
        if res is None:
            sub = pts if len(pts) <= 50000 else pts[np.random.default_rng(0).choice(len(pts), 50000, replace=False)]
            res = 1.5 * float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
        lo = pts.min(axis=0)
        idx = np.floor((pts - lo) / res).astype(int)
        occ = np.zeros(idx.max(axis=0) + 1, dtype=bool)
        occ[tuple(idx.T)] = True
        occ = np.pad(occ, pad, mode="constant", constant_values=False)   # pad BEFORE closing so edges don't erode
        if closing > 0:
            occ = ndi.binary_closing(occ, structure=ndi.generate_binary_structure(d, 1), iterations=closing)
        if fill_holes:
            occ = ndi.binary_fill_holes(occ)
        origin = lo + 0.5 * res - pad * res
        return cls(cls._signed_distance(occ, np.full(d, res)), np.full(d, res), origin)

    # ------------------------------------------------------------------ evaluation
    def _on(self, device):
        key = str(device)
        if key not in self._cache:
            vol = torch.as_tensor(self._sdf).permute(*range(self.d - 1, -1, -1))[None, None].contiguous().to(device)
            lo = torch.as_tensor(self.origin, device=device)
            hi = lo + (torch.tensor(self.shape, device=device, dtype=lo.dtype) - 1) * torch.as_tensor(self.spacing, device=device)
            self._cache[key] = (vol, lo, hi)
        return self._cache[key]

    def sdf_at(self, x):
        """Signed distance at physical points x (N, d). Differentiable w.r.t. x."""
        vol, lo, hi = self._on(x.device)
        g = 2.0 * (x.to(vol.dtype) - lo) / (hi - lo) - 1.0                      # normalised to [-1, 1]
        f = F.grid_sample(vol, g.reshape((1, -1) + (1,) * (self.d - 1) + (self.d,)),
                          mode="bilinear", padding_mode="border", align_corners=True)
        return f.reshape(-1).to(x.dtype)

    def __call__(self, x, lamb):
        return (0.5 * (1.0 + torch.tanh(self.sdf_at(x) / lamb))).reshape(-1, 1)


class NNetSupport(SupportMask):
    """Wraps the existing BoundaryClassifier (3D only): weight = 0.5 * (1 + tanh(f(x) / lamb))."""

    def __init__(self, nnetFile):
        import xmodmap
        self.bc = xmodmap.preprocess.BoundaryClassifier(param_file=nnetFile)

    def __call__(self, x, lamb):
        return self.bc.predict(1.0 / lamb, x).reshape(-1, 1)