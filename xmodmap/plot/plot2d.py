import numpy as np
from matplotlib.collections import EllipseCollection

from .utils import (
    default_radius_scale,
    feature_to_scalar,
    to_numpy,
    unpack_trajectory,
    weight_to_radius,
)


def _disc_collection(ax, pts, w, c, radius_scale, cmap, clim):
    d = 2 * weight_to_radius(w, radius_scale)
    col = EllipseCollection(
        d,
        d,
        np.zeros_like(d),
        offsets=pts,
        units="xy",
        offset_transform=ax.transData,
        array=c,
        cmap=cmap,
        clim=clim,
        alpha=0.7,
    )
    ax.add_collection(col)
    return col


def plot_registration_2d(
    S,
    w_S,
    zeta_S,
    T,
    w_T,
    zeta_T,
    trajectory=None,
    dims=(0, 1),
    radius_scale=None,
    cmap="tab10",
    figsize=(15, 5),
    show=True,
):
    """
    Plot source, target and deformed source particles in 2D, with a slider on time.

    Each particle is a disc: center = position, radius ~ sqrt(weight), color = feature (zeta).

    Parameters
    ----------
    S, T : (n, d) / (m, d) positions of source and target.
    w_S, w_T : (n,) / (m,) weights.
    zeta_S, zeta_T : (n,) / (n, k) features (k > 1, e.g. one-hot, is colored by argmax).
    trajectory : list of (px, pw, qx, qw) tuples as returned by `Shooting.__call__`
        (one per time step). Its qx / qw are used for the deformed source.
        If None, only the source and target are shown (no slider).
    dims : the two coordinates to display (for d > 2).
    radius_scale : multiplier of sqrt(weight); default makes the biggest disc ~1.5% of the extent.

    Returns the figure (the slider is kept alive on `fig._slider`).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    S, T = to_numpy(S)[:, list(dims)], to_numpy(T)[:, list(dims)]
    wS, wT = to_numpy(w_S).reshape(-1), to_numpy(w_T).reshape(-1)
    cS, cT = feature_to_scalar(zeta_S), feature_to_scalar(zeta_T)

    if trajectory is not None:
        Q, W = unpack_trajectory(trajectory, dims)
    else:
        Q, W = [S], [wS]

    if radius_scale is None:
        radius_scale = default_radius_scale([S, T] + Q, [wS, wT] + W)
    clim = (min(cS.min(), cT.min()), max(cS.max(), cT.max()))

    pts = np.concatenate([S, T] + Q)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.05 * (hi - lo).max()

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.15)

    kw = dict(radius_scale=radius_scale, cmap=cmap, clim=clim)
    _disc_collection(axes[0], S, wS, cS, **kw)
    _disc_collection(axes[1], T, wT, cT, **kw)
    deformed = _disc_collection(axes[2], Q[0], W[0], cS, **kw)

    for ax, title in zip(axes, ["Source", "Target", "Deformed source"]):
        ax.set_title(title)
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal")

    if trajectory is not None and len(Q) > 1:
        slider = Slider(
            fig.add_axes([0.3, 0.05, 0.4, 0.02]),
            "time step",
            0,
            len(Q) - 1,
            valinit=0,
            valstep=1,
            color="dimgray",
            track_color="lightgray",
            handle_style={"facecolor": "dimgray", "edgecolor": "white", "size": 12},
        )
        slider.label.set_color("dimgray")
        slider.valtext.set_color("dimgray")
        slider.ax.spines[:].set_visible(False)
        slider.vline.set_visible(False)  # valinit marker

        def update(val):
            k = int(slider.val)
            half_diameter = weight_to_radius(W[k], radius_scale)
            deformed.set_offsets(Q[k])
            # EllipseCollection stores half-axes in these private attributes
            deformed._widths = half_diameter
            deformed._heights = half_diameter
            axes[2].set_title(f"Deformed source (t = {k / (len(Q) - 1):.2f})")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(0)
        fig._slider = slider

    if show:
        plt.show()
    return fig
