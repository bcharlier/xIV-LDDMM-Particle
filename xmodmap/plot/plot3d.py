import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
from .utils import (
    default_radius_scale,
    pad_to_3d,
    feature_to_scalar,
    to_numpy,
    unpack_trajectory,
    weight_to_radius,
)


def _spheres(pv, pts, w, c, radius_scale, resolution):
    """Glyph mesh of spheres: center = position, radius ~ weight**(1/3), scalar = feature."""
    cloud = pv.PolyData(pts.astype(np.float32))
    cloud["radius"] = weight_to_radius(w, radius_scale, power=1 / 3)
    cloud["feature"] = c
    sphere = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
    return cloud.glyph(scale="radius", factor=1.0, geom=sphere, orient=False)


# Fragment code replacing the gaussian falloff of vtkPointGaussianMapper: the splat quad spans
# [-bound, bound]^2, so we cut a disc and shade it as a sphere impostor (headlight + ambient).
_SPHERE_SPLAT = """
//VTK::Color::Impl
float r2 = dot(offsetVCVSOutput.xy, offsetVCVSOutput.xy) / BOUND2;
if (r2 > 1.0) { discard; }
vec3 n = vec3(offsetVCVSOutput.xy / BOUND, sqrt(1.0 - r2));
float shade = 0.35 + 0.65 * max(dot(n, normalize(vec3(0.3, 0.4, 1.0))), 0.0);
float spec = 0.25 * pow(max(dot(n, normalize(vec3(0.3, 0.4, 1.0))), 0.0), 24.0);
diffuseColor = diffuseColor * shade + vec3(spec);
"""


def _cloud(pv, pts, w, c, radius_scale):
    """Point cloud for gaussian splatting: center = position, radius ~ weight**(1/3), scalar = feature."""
    cloud = pv.PolyData(pts.astype(np.float32))
    cloud["radius"] = weight_to_radius(w, radius_scale, power=1 / 3).astype(np.float32)
    cloud["feature"] = c
    return cloud


def plot_registration_3d(
    S,
    w_S,
    zeta_S,
    T,
    w_T,
    zeta_T,
    trajectory=None,
    pi_ST=None,
    radius_scale=None,
    cmap="tab10",
    style="sphere",
    resolution=8,
    target_opacity=0.05,
    window_size=(1500, 500),
    show=True,
    sphinx=True,
):
    """
    Plot source, target and deformed source particles in 3D with pyvista, with a slider on time.

    Each particle is a blob: center = position, radius ~ weight**(1/3), color = feature (zeta).

    Parameters
    ----------
    S, T : (n, d) / (m, d) positions of source and target; 2D data (d = 2) is shown in the z = 0 plane, viewed along z (x-y plane).
    w_S, w_T : (n,) / (n, 1) weights.
    zeta_S, zeta_T : (n,) / (n, k) features (k > 1, e.g. one-hot, is colored by argmax).
    trajectory : list of (px, pw, qx, qw) tuples as returned by `Shooting.__call__`
        (one per time step). Its qx / qw are used for the deformed source.
        If None, only the source and target are shown (no slider).
    radius_scale : multiplier of weight**(1/3); default makes the biggest particle ~1.5% of the extent.
    style : "gaussian" (soft gaussian splats, fast) or "sphere" (opaque tessellated spheres).
    resolution : sphere tessellation, only used for style="sphere".
    target_opacity : opacity of the target overlaid on the deformed source; a checkbox toggles it.
    show : if False, the plotter is returned without opening a window (e.g. for off-screen use).

    Returns the `pyvista.Plotter`.
    """
    import matplotlib
    import pyvista as pv

    is_2d = to_numpy(S).shape[1] < 3
    S, T = pad_to_3d(S), pad_to_3d(T)
    wS, wT = to_numpy(w_S).reshape(-1), to_numpy(w_T).reshape(-1)
    cS, cT = feature_to_scalar(zeta_S), feature_to_scalar(zeta_T)
    cS_mapped = feature_to_scalar(zeta_S @ pi_ST) if pi_ST is not None else cS
    cur = {"c": cS, "k": 0}   # color of the deformed source / current time step


    if trajectory is not None:
        Q, W = unpack_trajectory(trajectory, range(to_numpy(trajectory[0][2]).shape[1]))
        Q = [pad_to_3d(q) for q in Q]
    else:
        Q, W = [S], [wS]

    if radius_scale is None:
        radius_scale = default_radius_scale([S, T] + Q, [wS, wT] + W, power=1 / 3)

    if sphinx:
        style = "sphere"

    plotter = pv.Plotter(
        shape=(1, 3),
        window_size=list(window_size),
        off_screen=sphinx,
    )

    if style not in ("gaussian", "sphere"):
        raise ValueError(f"style must be 'gaussian' or 'sphere', got {style!r}")
    gaussian = style == "gaussian"


    K = int(max(cS.max(), cT.max())) + 1
    base = matplotlib.colormaps["tab20" if K > 10 else "tab10"]
    cmap_d = ListedColormap([base(i % base.N) for i in range(K)])   # exactly K colors
    clim = (-0.5, K - 0.5)
    kw = dict(scalars="feature", cmap=cmap_d, clim=clim, n_colors=K, show_scalar_bar=False)


    def build(pts, w, c):
        if gaussian:
            return _cloud(pv, pts, w, c, radius_scale)
        return _spheres(pv, pts, w, c, radius_scale, resolution)

    def add(mesh, **extra):
        if not gaussian:
            return plotter.add_mesh(mesh, **kw, **extra)
        actor = plotter.add_mesh(mesh, style="points_gaussian", emissive=False, render_points_as_spheres=False, **kw, **extra)
        actor.mapper.scale_array = "radius"
        mapper = actor.mapper
        mapper.SetBoundScale(1.0)
        mapper.SetSplatShaderCode(_SPHERE_SPLAT.replace("BOUND2", "1.0").replace("BOUND", "1.0"))
        return actor

    plotter.subplot(0, 0)
    plotter.add_text("Source", font_size=10)
    add(build(S, wS, cS))
    plotter.subplot(0, 1)
    plotter.add_text("Target", font_size=10)
    add(build(T, wT, cT))
    plotter.subplot(0, 2)
    deformed = build(Q[0], W[0], cur["c"])
    add(deformed)
    overlay = add(build(T, wT, cT), opacity=target_opacity, pickable=False)

    # an invisible box on the union of all the bounds gives the three views the same framing
    pts = np.concatenate([S, T] + Q)
    margin = 2 * radius_scale * max(w.max() for w in [wS, wT] + W) ** (1 / 3)
    box = pv.Box(bounds=np.ravel(np.stack([pts.min(0) - margin, pts.max(0) + margin], axis=1)))
    for col in range(3):
        plotter.subplot(0, col)
        plotter.add_mesh(box, opacity=0.0, pickable=False)
        plotter.enable_parallel_projection()
        plotter.reset_camera()
        if is_2d:
            plotter.view_xy()
    plotter.link_views()

    title = plotter.add_text("", position="upper_left", font_size=10)

    def update(value):
        k = int(round(value)); cur["k"] = k
        # in-place update keeps the actor, so the renderer picks the new geometry up
        new = build(Q[k], W[k], cur["c"])
        if gaussian:
            deformed.points = new.points
            deformed["radius"] = new["radius"]
            deformed["feature"] = new["feature"]
            # pyvista colors through a precomputed "__rgba__" array, so refresh it too
            lo, hi = clim
            deformed["__rgba__"] = (cmap_d((new["feature"] - clim[0]) / K) * 255).astype(np.uint8)
            deformed.Modified()
        else:
            deformed.copy_from(new)
        title.SetText(2, f"Deformed source (t = {k / max(len(Q) - 1, 1):.2f})")
        plotter.render()

    plotter.subplot(0, 2)
    plotter.add_checkbox_button_widget(
        lambda visible: (overlay.SetVisibility(bool(visible)), plotter.render()),
        value=True,
        position=(10, 10),
        size=30,
    )
    if pi_ST is not None:
        def toggle_map(on):
            cur["c"] = cS_mapped if on else cS
            update(cur["k"])
        plotter.add_checkbox_button_widget(toggle_map, value=False, position=(10, 50), size=30)
        plotter.add_text("map by pi_ST", position=(50, 55), font_size=8)
    plotter.add_text("target", position=(50, 15), font_size=8)
    if trajectory is not None and len(Q) > 1:
        plotter.add_slider_widget(
            update,
            rng=[0, len(Q) - 1],
            value=0,
            title="time step",
            fmt="{:.0f}",
            interaction_event="always",
            pointa=(0.3, 0.07),
            pointb=(0.9, 0.07),
            style="modern",
            color="dimgray",
            title_height=0.02,
            slider_width=0.02,
            tube_width=0.004,
        )
    update(0)

    if show:
        plotter.show(auto_close=False)

    return plotter
