import numpy as np
import torch
from scipy.spatial import cKDTree


class LocalAreaChange:
    """
    Local area (2D) / volume (3D) change of a deformation, estimated from each particle's k nearest neighbours.

    With q the particle positions before the deformation and q' after it, the local deformation gradient of particle i
    is the least-squares fit over its neighbours j in N(i):

        F_i = argmin_F  sum_j | (q'_j - q'_i) - F (q_j - q_i) |^2
            = [ sum_j dq'_ij dq_ij^T ] [ sum_j dq_ij dq_ij^T ]^{-1},          J_i = det F_i

    The neighbour sets and the matrices [sum_j dq dq^T]^{-1} come from the reference positions only and are computed once.
    Each call then needs just the deformed positions q' (which the loss already has), so no extra shooting is needed, and
    it is differentiable in q'. Use unit-box positions: det is unchanged by a uniform rescaling.

    The estimate is noisy per particle (random sampling), so use it through the mean, which is what penalty() does:

        g = mean_i log J_i      (g < 0: the source shrank, exp(g) is the geometric-mean area ratio)

    penalty() is a hinge on g, zero while  -tol_shrink <= g <= tol_expand  and quadratic outside, so ordinary local
    compression and expansion that cancel out are not constrained. The sampling is assumed roughly uniform over the
    source, as it is for the downsampled sections.
    """

    def __init__(self, points, k=10, eps=1e-6, det_floor=1e-3):
        pts = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
        pts = pts.astype(np.float64)
        n, d = pts.shape
        k = min(int(k), n - 1)
        if k < d + 1:
            raise ValueError(f"need at least {d + 1} neighbours to fit a {d}D deformation gradient, got k={k}")
        idx = cKDTree(pts).query(pts, k=k + 1)[1][:, 1:]                    # drop the point itself
        dq = pts[idx] - pts[:, None, :]                                     # (n, k, d)
        A = np.einsum("nka,nkb->nab", dq, dq)                               # sum_j dq dq^T
        tr = np.trace(A, axis1=1, axis2=2) / d
        tr = np.where(tr > 0, tr, 1.0)                                      # duplicated points: fall back to identity
        A = A + (eps * tr)[:, None, None] * np.eye(d)                       # keeps collinear neighbourhoods invertible
        self.d = d
        self.k = k
        self.det_floor = det_floor
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.dq_ref = torch.as_tensor(dq)
        self.Ainv = torch.as_tensor(np.linalg.inv(A))
        self._cache = {}

    def _on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            self._cache[key] = (self.idx.to(device),
                                self.dq_ref.to(device=device, dtype=dtype),
                                self.Ainv.to(device=device, dtype=dtype))
        return self._cache[key]

    def log_det(self, q):
        """log det F_i for every particle, from the deformed positions q (n, d). Differentiable."""
        idx, dq_ref, Ainv = self._on(q.device, q.dtype)
        dq_new = q[idx] - q[:, None, :]                                     # (n, k, d)
        B = torch.einsum("nka,nkb->nab", dq_new, dq_ref)                    # sum_j dq' dq^T
        det = torch.linalg.det(B @ Ainv)
        return torch.log(det.clamp_min(self.det_floor))

    def global_log_ratio(self, q):
        """g = mean log det F: log of the (geometric-mean) area / volume ratio of the deformation."""
        return self.log_det(q).mean()

    def penalty(self, q, tol=0.3, local_tol=None):
        """
        tol       : allowed |g|, or a (tol_shrink, tol_expand) pair. 0.3 allows the area to change by about -26% / +35%.
        local_tol : optional, also penalise individual particles with |log J_i| > local_tol (0 / None = off).
        Returns (penalty, g).
        """
        ld = self.log_det(q)
        g = ld.mean()
        tol_shrink, tol_expand = tol if isinstance(tol, (tuple, list)) else (tol, tol)
        pen = torch.relu(-g - tol_shrink) ** 2 + torch.relu(g - tol_expand) ** 2
        if local_tol:
            pen = pen + (torch.relu(ld.abs() - local_tol) ** 2).mean()
        return pen, g