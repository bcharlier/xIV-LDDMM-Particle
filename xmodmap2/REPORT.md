# xmodmap2: decoupled deformation modes, closed-form rigid flows, and timings

This report covers four things:

1. The structure of `xmodmap2`, a reorganized copy of `xmodmap`.
2. What the coefficient `cS` of `xmodmap` really does, and how `xmodmap2` makes it explicit.
3. The derivation of the closed-form flows used by the rigid and affine modes.
4. Running-time and accuracy measurements for each mode.

`xmodmap` itself is unchanged. The new example script is
`examples/rotation/framework_script_2d_rotation_yi.py` and the benchmark is
`examples/rotation/benchmark_modes_yi.py`. The tests are in `tests/test_xmodmap2_deformation.py`.

---

## 1. Summary

- **One `Deformation` object replaces `Hamiltonian` and `Shooting`.**
  A string selects the mode: `"translation"`, `"rotation"`, `"scaling"`, `"rigid"`, `"similarity"`,
  `"affine"`, `"non-rigid"`, `"all-mixed"`, or any `+` combination such as `"non-rigid+rigid"`.
  The Hamiltonian is a property of the deformation, and shooting is a method of the Hamiltonian.
- **`"all-mixed"` reproduces `xmodmap`.** The Hamiltonian values are identical and the shot
  states agree to float32 round-off (relative error about 1e-7).
- **`cS` is now `cNR`, the cost of the non-rigid mode.** It no longer hides inside the kernel
  normalization. Scaling gets its own cost `cScale`, which `xmodmap` hard-coded to 1.
- **Every finite-dimensional mode has an exact flow.** Translation, rotation, scaling, rigid,
  rotation with isotropic scaling, and full affine need no time stepping. Only rotation combined
  with anisotropic scaling in 3D, and any mode containing the non-rigid part, are integrated
  numerically.
- **Speed.** A closed-form deformation step is 15 to 25 times faster than integrating the same
  mode, and it removes the time-discretization error. On the rotation example with 2000 points,
  `rigid` recovers the 0.6 rad rotation in 1.8 s. `all-mixed` reaches the same angle in 407 s.
  Section 5 has the tables.

---

## 2. Code structure

### 2.1 Package layout

```
xmodmap2/
├── __init__.py              same dtype setup as xmodmap; exposes distance, deformation, model
├── REPORT.md                this report
├── deformation/             ← redesigned
│   ├── __init__.py
│   ├── Deformation.py       Deformation class, mode-string parser
│   └── hamiltonian/
│       ├── base.py          Hamiltonian (mother class): Hamilton's equations, integrators, shooting
│       ├── kernels.py       KeOps formulas, kernel normalization N_sigma (old Ucoeff without cS)
│       ├── nonrigid.py      NonRigidHamiltonian
│       ├── finitedim.py     FiniteDimHamiltonian + Translation / Rotation / Scaling / GeneralLinear,
│       │                    closed-form flow
│       └── mixed.py         MixedHamiltonian (sum of modes sharing one momentum)
├── model/                   SingleModality / CrossModality / CrossModalityBoundary take a Deformation
├── distance/  optimizer/  preprocess/  io/  utils.py    copied from xmodmap, imports renamed
```

The old `deformation/Hamiltonian.py`, `Shooting.py`, `Ucoeff.py` and `control/` are absorbed into
`deformation/hamiltonian/`.

### 2.2 Class hierarchy

```
Hamiltonian                      base.py      H, velocity, divergence (abstract)
│                                             system()          Hamilton's equations by autograd (+ passive points)
│                                             shoot()           trajectory, closed form if available
│                                             endpoint()        state at t = 1 only
│                                             shoot_backward()  former ShootingBackwards
│                                             integrate()       euler / ralston (default) / rk4
├── NonRigidHamiltonian          nonrigid.py  multi-scale Gauss or Laplace RKHS, cost cNR
├── FiniteDimHamiltonian         finitedim.py v(x) = L (x - c) + tau, exact flow_trajectory(), flow_map()
│   ├── TranslationHamiltonian                cost cT
│   ├── RotationHamiltonian                   cost cA
│   ├── ScalingHamiltonian                    cost cScale, axis groups follow dimEff / single
│   └── GeneralLinearHamiltonian              cost cL (any d x d matrix)
└── MixedHamiltonian             mixed.py     sum of components; closed form when the sum is solvable

Deformation                      Deformation.py   mode string -> Hamiltonian; weight (gamma); energy(), energies(),
                                                  shoot(), __call__ (endpoint), transform(), inverse_transform(), flow_map()
```

Every parameter of a mode is stored on its Hamiltonian instance: scales, coefficients, dimension,
number of time steps and integrator. Shooting therefore needs no extra arguments and lives on the
Hamiltonian. A child class with an exact solution sets `has_closed_form = True` and implements
`flow_trajectory`. The generic `shoot` then skips the integrator.

### 2.3 Modes

| mode string | components (summed, shared momentum) | flow |
|---|---|---|
| `translation` | translation | closed form |
| `rotation` | rotation about the barycenter | closed form |
| `scaling` | diagonal scaling about the barycenter | closed form |
| `rigid` | rotation + translation | closed form |
| `similarity` | rotation + translation + scaling (the linear part of `xmodmap`) | closed form if the scaling is isotropic (2D, or 3D with `single=True`), numerical otherwise |
| `affine` | general linear matrix + translation | closed form |
| `linear` | general linear matrix | closed form |
| `non-rigid` | RKHS velocity field | numerical |
| `all-mixed` | non-rigid + rotation + translation + scaling (= `xmodmap`) | numerical |
| `a+b+...` | any combination, e.g. `non-rigid+rigid`, `affine+non-rigid` | closed form only if every part is finite-dimensional and the sum is solvable |

Synonyms such as `nonrigid`, `lddmm` and `mixed` are accepted. Combining `linear` or `affine`
with `rotation` or `scaling` is rejected because the general linear mode already contains them.

### 2.4 Coefficients

| xmodmap2 | xmodmap | mode | default |
|---|---|---|---|
| `cA` | `cA` | rotation | 1 |
| `cT` | `cT` | translation | 1 |
| `cScale` | none, fixed to 1 | scaling | 1 |
| `cNR` | `cS` | non-rigid | 10 |
| `cL` | none | general linear | 1 |
| `weight` | `hamiltonian.weight` | gamma, weight of H in the loss | 1 |

Passing `cS=` raises an error that explains the renaming.

### 2.5 Using it

Old (`xmodmap`):

```python
hamiltonian = xmodmap.deformation.Hamiltonian(sigmaRKHS, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)
hamiltonian.weight = gamma
shooting = xmodmap.deformation.Shooting(sigmaRKHS, Stilde, cA=cA, cS=cS, cT=cT, dimEff=dimEff, single=single)
loss = xmodmap.model.SingleModality(hamiltonian, shooting, dataloss)
px1, pw1, qx1, qw1 = shooting(px, pw, qx, qw)[-1]
```

New (`xmodmap2`):

```python
deformation = xmodmap2.deformation.Deformation("all-mixed", Stilde, sigma=sigmaRKHS,
                                               cA=cA, cT=cT, cScale=1.0, cNR=10.0,
                                               dimEff=dimEff, single=single)
deformation.weight = gamma
loss = xmodmap2.model.SingleModality(deformation, dataloss)

px1, pw1, qx1, qw1 = deformation(px, pw, qx, qw)               # final state
traj = deformation.hamiltonian.shoot(px, pw, qx, qw)            # list of states, t = k/nt
X1, wX1 = deformation.transform(px, pw, qx, qw, X, wX)          # push other points (former ShootingGrid)
X0, wX0 = deformation.inverse_transform(px1, pw1, qx1, qw1, X)  # pull back (former ShootingBackwards)
deformation.energies(px, pw, qx, qw)   # {'non-rigid': ..., 'rotation': ..., 'translation': ..., 'scaling': ...}
Phi, b = Deformation("rigid", Stilde).flow_map(px, pw, qx, qw)  # explicit x -> Phi x + b (closed-form modes)
```

The example script selects the mode on the command line:

```
python examples/rotation/framework_script_2d_rotation_yi.py --mode rigid
python examples/rotation/framework_script_2d_rotation_yi.py --mode all-mixed --steps 300
python examples/rotation/framework_script_2d_rotation_yi.py --mode "non-rigid+rigid" --npoints 2000
```

At the end it prints the energy of each mode and the recovered rotation and translation. It compares
both against the ground truth. For numerically integrated modes the rotation comes from a weighted
Procrustes fit of the deformed source.

### 2.6 Problems found in the original code

Porting turned up the problems below. All fixes are in `xmodmap2` and the new script only, and
`xmodmap` is untouched.

- **Data format.** The data file holds a dict of mesh fields with 3D points and no features. The
  original script unpacks it as `(S, nu_S)`, which fails. The new script reads the dict and uses
  unit masses. The `nu_X` features exist only in the `.vtk` file.
- **Rotation of 3D points.** The original script applies a 2×2 rotation to N×3 points. If the points
  were 2D, the step meant to flatten z would zero the y coordinate instead. The new script
  reduces to 2D first.
- **Missing output folder.** `sm.pt` was saved before the output folder existed.
- **3D data with an effective dimension of 2.** `xmodmap.deformation.control.affine.getATauAlpha`
  crashes with `IndexError` because it indexes the 1-D barycenter as `xc[:, :2]`. The test suite
  records this as an expected failure of `xmodmap`.
- **Default preconditioner.** `Model.precond = lambda *x: x` becomes a bound method returning a
  tuple, so any model built without `precond` crashes. `precondWeights` was also undefined.
- **Boundary model.** `CrossModalityBoundary.__init__` never called `Model.__init__`, so `self.logs`
  was missing and the first loss evaluation crashed.
- **Laplace kernel, not fixed.** With nonzero mass momenta, the Laplace kernel Hamiltonian has a
  $(d-1)/R$ term at $i=j$ where $R = 10^{-4}$. That term is huge. The kernel is ported as is, and
  its normalization now uses the Laplace kernel itself. `xmodmap` defined this kernel but never
  called it.

---

## 3. What `cS` does, made explicit

### 3.1 Before

In `xmodmap` the non-rigid kernel of scale $\sigma$ was divided by

$$
\texttt{uCoeff}_\sigma = c_S\,N_\sigma,\qquad
N_\sigma = \frac{1}{N^2\sigma^2}\sum_{i,j} K_\sigma(\tilde S_i,\tilde S_j),
$$

and the total Hamiltonian was

$$
H = \sum_\sigma \frac{1}{c_S N_\sigma}\,H_\sigma(p,q)
  + \frac{c_A}{2}\lVert A\rVert_F^2 + \frac{c_T}{2}\lvert\tau\rvert^2 + \frac{1}{2}\lVert B\rVert_F^2 ,
$$

where $H_\sigma$ is the Gaussian-kernel Hamiltonian of scale $\sigma$, $A$ the rotation, $\tau$ the
translation and $B$ the diagonal scaling. Throughout, $q$ collects the positions and masses and
$p=(p_x,p_w)$ the position and mass momenta, with $p_{i,x}$ and $p_{i,w}$ the momenta of particle $i$.

Three things made the effect of `cS` hard to read:

- **The name.** `cS` reads like a scaling cost, but it is the non-rigid cost. The scaling cost was a
  hidden constant 1.
- **The mixing.** `cS` was multiplied into a data-dependent normalization, so the code never showed
  it as a plain coefficient.
- **The hidden constant.** The scaling cost stayed at 1. So multiplying `cS`, `cA` and `cT` by the
  same factor was not equivalent to changing gamma, because it also moved motion into or out
  of the scaling mode.

### 3.2 After

`NonRigidHamiltonian` stores the data-dependent part and the user coefficient separately:

$$
\omega_\sigma = \frac{1}{c_{NR}\,N_\sigma},\qquad
u(x) = \sum_\sigma \omega_\sigma \sum_j K_\sigma(x,q_j)\Big(p_{j,x} + p_{j,w}\,w_j\,\frac{x-q_j}{\sigma^2}\Big)\quad\text{(Gauss kernel)} .
$$

Here $w_j$ is the mass of particle $j$, and $p_{j,x}$ and $p_{j,w}$ are its position and mass momenta. `self.normalization` holds
$N_\sigma$, `self.scale_weights` holds $\omega_\sigma$ and `self.cNR` holds $c_{NR}$. The value $N_\sigma$ is a mean
of kernel values over pairs, scaled by $1/\sigma^2$. It depends on the extent of the shape relative to
$\sigma$ and hardly on the number of points. It only brings the scales to the same order of
magnitude.

### 3.3 How each coefficient acts

Each mode $k$ has a control $u_k$, a velocity $v_k = \mathcal L_k u_k$ and a cost
$\tfrac{c_k}{2}\lVert u_k\rVert^2$. Its reduced Hamiltonian is

$$
H_k(p,q) = \max_{u_k}\;\langle p, \mathcal L_k u_k\rangle - \frac{c_k}{2}\lVert u_k\rVert^2
         = \frac{1}{2c_k}\,\lVert\mathcal L_k^{\ast} p\rVert^2,\qquad u_k^{\ast} = \frac{1}{c_k}\mathcal L_k^{\ast} p .
$$

- **Each coefficient is an inverse gain.** For a given momentum, mode $k$ moves the points by an
  amount proportional to $1/c_k$. In 2D this gives, for example, a rotation angle
  $\theta = \frac{1}{2c_A}\sum_i (q_i-c)\times p_{i,x}$ and a translation $\tau = \frac{1}{c_T}\sum_i p_{i,x}$.
- **Mixed modes share one momentum.** The optimizer pays $c_k\lVert u_k\rVert^2/2$ per mode, so it puts the
  motion where it is cheapest. Raising $c_{NR}$ shifts motion from the non-rigid part to the
  rigid and scaling parts. As $c_{NR}\to\infty$, `all-mixed` tends to `similarity`. This is the
  "10000 for rigid" trick in the original script. `xmodmap2` can instead select `"rigid"` or
  `"similarity"` directly.
- **Only the ratios matter.** Multiply every $c_k$ by $\lambda$. Then $H^{(\lambda)}(p)=H(p)/\lambda$,
  and with $p=\lambda p'$ Hamilton's equations become those of $H$ in $p'$, since $H$ is quadratic in
  $p$. Writing $q(1;p)$ for the final positions and masses, the loss becomes

  $$
  \gamma H^{(\lambda)}(p) + D\big(q(1;p)\big) = \lambda\gamma\,H(p') + D\big(q(1;p')\big).
  $$

  A common factor on all coefficients is therefore exactly a change of gamma. The ratios
  $c_{NR}:c_A:c_T:c_{\mathrm{scale}}$ decide how motion splits between modes. The product
  $\gamma\,c_{NR}$ decides how strongly the non-rigid part is regularized against the data term.
  The test `test_common_rescaling_of_coefficients_equals_rescaling_gamma` checks this identity.
- **Energy per mode.** `deformation.energies(...)` returns each $H_k$. After an optimization it
  shows which mode carried the motion.

---

## 4. Closed-form flows of the finite-dimensional modes

### 4.1 Setting

The particles are indexed by $i=1..N$. Each has a position $q_i\in\mathbb R^d$ (`qx`), a mass
$w_i>0$ (`qw`), a position momentum $p_{i,x}\in\mathbb R^d$ (`px`) and a mass momentum $p_{i,w}\in\mathbb R$ (`pw`). Define

$$
\begin{aligned}
&W=\sum_i w_i,\qquad c=\frac1W\sum_i w_i q_i,\qquad y_i=q_i-c,\qquad \lambda_i=\frac{w_i}{W},\\
&P_x=\sum_i p_{i,x},\qquad P_w=\sum_i p_{i,w}\,w_i,\qquad M=\sum_i p_{i,x}\,y_i^\top \quad (d\times d\ \text{matrix}).
\end{aligned}
$$

Every finite-dimensional mode, and every sum of them, produces a velocity field

$$
v(x) = L\,(x-c) + \tau,\qquad \operatorname{div} v = \operatorname{tr} L,
$$

so that $\dot q_i = L y_i + \tau$ and $\dot w_i = (\operatorname{tr}L)\,w_i$. The pairing between
momenta and velocity is

$$
\sum_i \langle p_{i,x}, v(q_i)\rangle + p_{i,w}\,w_i \operatorname{div}v(q_i)
 = \langle L,\; M + P_w I\rangle_F + \langle \tau, P_x\rangle .
$$

### 4.2 Controls and Hamiltonians

Maximizing the pairing minus each quadratic cost gives the following. Here $E_g$ is the 0/1 diagonal
mask of axis group $g$, and $k_g$ is the number of axes in $g$.

| mode | optimal control | $H$ |
|---|---|---|
| translation | $\tau = P_x/c_T$ | $\tfrac{c_T}{2}\lvert\tau\rvert^2$ |
| rotation | $A = \dfrac{M-M^\top}{2c_A}$ | $\tfrac{c_A}{2}\lVert A\rVert_F^2$ |
| scaling | $B=\sum_g a_g E_g$ with $a_g=\dfrac{\langle E_g,M\rangle + k_g P_w}{c_{\mathrm{scale}}\,k_g}$ | $\tfrac{c_{\mathrm{scale}}}{2}\lVert B\rVert_F^2$ |
| general linear | $L = \dfrac{M+P_w I}{c_L}$ | $\tfrac{c_L}{2}\lVert L\rVert_F^2$ |

With $c_{\mathrm{scale}}=1$ these are exactly `getATauAlpha` of `xmodmap`, branch by branch. The axis
groups are (x, y) in 2D, (x, y) for 3D data with an effective dimension of 2, (x, y, z) for
`single=True`, and (x, y) plus (z) otherwise. A sum of modes has $L=\sum_k L_k$, $\tau=\sum_k\tau_k$
and $H=\sum_k H_k$.

### 4.3 Hamilton's equations

Every finite-dimensional Hamiltonian depends on the positions and masses only through the moments
$M$ and $P_w$. Their derivatives are

$$
\mathrm dM = \big(p_{i,x} - \lambda_i P_x\big)\,\mathrm dq_i^\top,
\qquad
\frac{\partial M}{\partial w_i} = -\frac{P_x\,y_i^\top}{W},\qquad \frac{\partial P_w}{\partial w_i}=p_{i,w} .
$$

Here $\mathrm dM$ is the first-order change of $M$ when $q_i$ moves by $\mathrm dq_i$ and every other
variable is held fixed. Both $p_{i,x}$ and $\mathrm dq_i$ are column vectors in $\mathbb R^d$, so
$\big(p_{i,x}-\lambda_iP_x\big)\,\mathrm dq_i^\top$ is a $d\times d$ outer product, like $M$. Moving $q_i$
changes $y_i$, but it also moves the barycenter $c$, and that shifts every other $y_j$. The term
$-\lambda_iP_x\,\mathrm dq_i^\top$ is this barycenter effect.

#### Proof of the derivatives

In Hamiltonian mechanics the momenta are independent variables. So $p_{j,x}$ and $p_{j,w}$ stay
fixed when we differentiate with respect to $q_i$ or $w_i$. Written as explicit functions of the
positions and masses, the quantities involved are

$$
\begin{aligned}
W(w) &= \sum_j w_j, & c(q,w) &= \frac{1}{W}\sum_j w_j\,q_j,\\
M(q,w) &= \sum_j p_{j,x}\,\big(q_j-c\big)^\top, & P_w(w) &= \sum_j p_{j,w}\,w_j .
\end{aligned}
$$

The total position momentum $P_x=\sum_j p_{j,x}$ depends on neither $q$ nor $w$, and $P_w$ does not
depend on $q$.

**Step 1: the barycenter against $q_i$.** $W$ does not involve $q_i$. In the sum $\sum_j w_jq_j$,
only the term $j=i$ involves $q_i$, and its change is $w_i\,\mathrm dq_i$. Hence

$$
\mathrm dc = \frac{w_i}{W}\,\mathrm dq_i = \lambda_i\,\mathrm dq_i .
$$

**Step 2: $M$ against $q_i$.** Differentiate the sum defining $M$ term by term. For $j\neq i$ only $c$
moves, so $\mathrm d(q_j-c)=-\mathrm dc$. For $j=i$ both move, so $\mathrm d(q_i-c)=\mathrm dq_i-\mathrm dc$.
Collecting the terms and using Step 1:

$$
\begin{aligned}
\mathrm dM &= p_{i,x}\,\mathrm dq_i^\top \;-\; \sum_j p_{j,x}\,\mathrm dc^\top \\
           &= p_{i,x}\,\mathrm dq_i^\top \;-\; P_x\,\big(\lambda_i\,\mathrm dq_i\big)^\top \\
           &= \big(p_{i,x}-\lambda_i P_x\big)\,\mathrm dq_i^\top .
\end{aligned}
$$

**Step 3: the barycenter against $w_i$.** Now both the numerator $\sum_j w_jq_j$ and the
denominator $W$ depend on $w_i$, with $\partial W/\partial w_i=1$ and
$\partial\big(\sum_j w_jq_j\big)/\partial w_i=q_i$. By the quotient rule,

$$
\frac{\partial c}{\partial w_i}
 = \frac{q_i}{W} - \frac{\sum_j w_j\,q_j}{W^2}
 = \frac{q_i - c}{W}
 = \frac{y_i}{W}.
$$

**Step 4: $M$ against $w_i$.** No position $q_j$ depends on $w_i$, so in each term of $M$ only $c$
moves:

$$
\frac{\partial M}{\partial w_i}
 = -\sum_j p_{j,x}\Big(\frac{\partial c}{\partial w_i}\Big)^\top
 = -\Big(\sum_j p_{j,x}\Big)\frac{y_i^\top}{W}
 = -\frac{P_x\,y_i^\top}{W}.
$$

**Step 5: $P_w$ against $w_i$.** $P_w$ is linear in the masses with fixed coefficients $p_{j,w}$, so
only the term $j=i$ survives, and $\partial P_w/\partial w_i=p_{i,w}$. This proves the three formulas.

#### From the derivatives to Hamilton's equations

Each finite-dimensional Hamiltonian has the form

$$
H = h\big(M,P_w,P_x\big),\qquad
h\big(M,P_w,P_x\big) = \max_{L,\,\tau}\;\Big[\langle L,\,M+P_wI\rangle_F + \langle\tau,P_x\rangle - \mathrm{cost}(L,\tau)\Big],
$$

where $L$ ranges over the matrices allowed by the mode, and the cost is the quadratic cost in the
table of 4.2. The bracket is linear in $(M,P_w,P_x)$ and strictly concave in the controls, so the
maximizer $(L,\tau)$ is unique. By the envelope theorem, the gradient of $h$ equals the gradient of
the bracket with the controls held at that maximizer. Since $\langle L,P_wI\rangle_F=P_w\operatorname{tr}L$, this gives

$$
\frac{\partial h}{\partial M}=L,\qquad \frac{\partial h}{\partial P_w}=\operatorname{tr}L,\qquad \frac{\partial h}{\partial P_x}=\tau .
$$

For a sum of modes, $h$ is the sum of the per-mode functions, so $L$ and $\tau$ are the summed controls.

The chain rule combines these gradients with Steps 2, 4 and 5. It also uses the identity
$\langle G,\,a\,b^\top\rangle_F=a^\top G\,b$ for a matrix $G$ and vectors $a,b$.

- **Against $q_i$.** $\mathrm dH=\langle L,\mathrm dM\rangle_F=\big(p_{i,x}-\lambda_iP_x\big)^\top L\,\mathrm dq_i$,
  so $\partial H/\partial q_i=L^\top\big(p_{i,x}-\lambda_iP_x\big)$.
- **Against $w_i$.** $\partial H/\partial w_i=\langle L,\partial M/\partial w_i\rangle_F+\operatorname{tr}L\;\partial P_w/\partial w_i$.
  With Steps 4 and 5 this is $\partial H/\partial w_i=-P_x^\top L\,y_i/W+(\operatorname{tr}L)\,p_{i,w}$.
- **Against $p_{i,x}$.** Moving $p_{i,x}$ changes $M$ by $\mathrm dp_{i,x}\,y_i^\top$ and $P_x$ by
  $\mathrm dp_{i,x}$. So $\mathrm dH=\langle L,\mathrm dp_{i,x}\,y_i^\top\rangle_F+\langle\tau,\mathrm dp_{i,x}\rangle$,
  and $\partial H/\partial p_{i,x}=L\,y_i+\tau$.
- **Against $p_{i,w}$.** Moving $p_{i,w}$ changes $P_w$ by $w_i\,\mathrm dp_{i,w}$, so
  $\partial H/\partial p_{i,w}=(\operatorname{tr}L)\,w_i$.

Hamilton's equations are $\dot q_i=\partial H/\partial p_{i,x}$, $\dot w_i=\partial H/\partial p_{i,w}$,
$\dot p_{i,x}=-\partial H/\partial q_i$ and $\dot p_{i,w}=-\partial H/\partial w_i$. Hence

$$
\boxed{
\begin{aligned}
\dot q_i &= L\,y_i + \tau, &\qquad \dot w_i &= (\operatorname{tr}L)\,w_i,\\
\dot p_{i,x} &= -L^\top\big(p_{i,x}-\lambda_i P_x\big), & \dot p_{i,w} &= -(\operatorname{tr}L)\,p_{i,w} + \frac{P_x^\top L\,y_i}{W}.
\end{aligned}}
$$

The translation Hamiltonian depends only on $P_x$, so it enters only $\dot q_i$.

### 4.4 Conserved quantities

The following hold along the flow, using $\sum_i w_i y_i = 0$:

1. $\dot P_x = -L^\top(P_x-\sum_i\lambda_iP_x)=0$. So $P_x$ and $\tau=P_x/c_T$ are constant.
2. $\dot W=(\operatorname{tr}L)W$. So every $\lambda_i=w_i/W$ is constant.
3. $\frac{\mathrm d}{\mathrm dt}\sum_i w_iq_i = (\operatorname{tr}L)\,Wc + W\tau$. So $\dot c=\tau$
   and $c(t)=c_0+t\tau$.
4. $\dot y_i = L\,y_i$.
5. $\dot P_w = \sum_i\big(-(\operatorname{tr}L)\,p_{i,w} + P_x^\top Ly_i/W\big)w_i + p_{i,w}(\operatorname{tr}L)\,w_i = 0$.
6. $\dot M = \sum_i \dot p_{i,x}\,y_i^\top + p_{i,x}\,\dot y_i^\top = -L^\top M + M L^\top = [M, L^\top]$.

The controls are linear in $(M,P_w)$, so items 5 and 6 give the dynamics of $L$. This is the
Euler–Arnold equation of the mode.

### 4.5 Dynamics of $L$, mode by mode

Write $\operatorname{skew}X=\tfrac12(X-X^\top)$ and $\operatorname{sym}X=\tfrac12(X+X^\top)$. For $A$ skew,
$[A,M]^\top=[A,M^\top]$, so $\operatorname{skew}[A,M]=[A,\operatorname{skew}M]$.

- **Rotation** ($L=A$, so $L^\top=-A$). We have $c_A\dot A = \operatorname{skew}[M,-A] = \operatorname{skew}[A,M] = [A,\operatorname{skew}M] = c_A[A,A]=0$.
  The rotation velocity $A$ is constant.
- **Scaling** ($L=B$ diagonal). We have
  $\frac{\mathrm d}{\mathrm dt}\langle E_g,M\rangle=\operatorname{tr}(E_gMB-E_gBM)=\operatorname{tr}(MBE_g-ME_gB)=0$,
  because diagonal matrices commute. With $\dot P_w=0$, $B$ is constant, whether isotropic or not.
- **Rotation plus scaling** ($L=A+B$). Here $\dot M=[M,B]+[A,M]$, and
  - $c_A\dot A=\operatorname{skew}[M,B]=[\operatorname{sym}M,\,B]$,
  - $\frac{\mathrm d}{\mathrm dt}\langle E_g,M\rangle=\langle [E_g,A]^\top, M\rangle$.

  Both vanish for every $M$ exactly when the scaling is isotropic, $B=bI$ with one group holding
  all $d$ axes. Then $L$ is constant. In 2D this always holds. In 3D with xy and z factors, or with
  z unscaled, $L$ changes along the flow. That combination is integrated numerically, and
  `test_rotation_plus_anisotropic_scaling_is_integrated_numerically` shows $L(1)\neq L(0)$.
- **Translation** does not affect $L$. It can be added to any of the cases above.
- **General linear** ($L=(M+P_w I)/c_L$). Here
  $\dot L = [M,L^\top]/c_L=[c_LL-P_w I,\,L^\top]/c_L=[L,L^\top]$, so $L$ is not constant in general.
  Split $L=S+K$ into symmetric and skew parts. Then $[L,L^\top]=[S+K,S-K]=2[K,S]$, which is
  symmetric. Hence

  $$
  \dot K=0,\qquad \dot S=2[K,S]\quad\Longrightarrow\quad S(t)=e^{2tK_0}S_0e^{-2tK_0},\qquad L(t)=K_0+e^{2tK_0}S_0e^{-2tK_0}.
  $$

  To solve $\dot\Phi=L(t)\Phi$, set $Z=e^{-2tK_0}\Phi$:

  $$
  \dot Z=-2K_0Z+e^{-2tK_0}\big(K_0+e^{2tK_0}S_0e^{-2tK_0}\big)\Phi=(S_0-K_0)Z=L_0^\top Z,
  $$

  so $\Phi(t)=e^{2tK_0}\,e^{tL_0^\top}$. This is the geodesic of the right-invariant Frobenius
  metric on the affine group. If $L_0$ is normal, it reduces to $e^{tL_0}$.

### 4.6 The explicit solution

Let $\Phi(t)$ solve $\dot\Phi=L(t)\Phi$ with $\Phi(0)=I$. Then:

- **Positions.** $y_i(t)=\Phi(t)y_i(0)$ and $c(t)=c_0+t\tau$.
- **Masses.** $w_i(t)=w_i(0)\exp\!\int_0^t\operatorname{tr}L=w_i(0)\det\Phi(t)$. In every case
  $\operatorname{tr}L(t)=\operatorname{tr}L_0$, so $\det\Phi(t)=e^{t\operatorname{tr}L_0}$.
- **Position momenta.** $r_i=p_{i,x}-\lambda_iP_x$ satisfies $\dot r_i=-L^\top r_i$, and
  $\frac{\mathrm d}{\mathrm dt}\Phi^{-\top}=-L^\top\Phi^{-\top}$. So $r_i(t)=\Phi(t)^{-\top}r_i(0)$.
- **Mass momenta.** We have
  $\frac{\mathrm d}{\mathrm dt}(\det\Phi\,p_{i,w})=\det\Phi\,\frac{P_x^\top L\,\Phi\,y_i(0)}{W_0\det\Phi}=\frac{P_x^\top\dot\Phi\,y_i(0)}{W_0}$,
  which integrates directly.

Altogether:

$$
\boxed{
\begin{aligned}
q_i(t) &= c_0 + t\,\tau + \Phi(t)\,\big(q_i(0)-c_0\big)\\
w_i(t) &= \det\Phi(t)\; w_i(0)\\
p_{i,x}(t) &= \lambda_i P_x + \Phi(t)^{-\top}\big(p_{i,x}(0)-\lambda_i P_x\big)\\
p_{i,w}(t) &= \frac{1}{\det\Phi(t)}\Big[p_{i,w}(0) + \frac{1}{W_0}\,P_x^\top\big(\Phi(t)-I\big)\big(q_i(0)-c_0\big)\Big]
\end{aligned}}
$$

The value of $\Phi$ depends on the mode, with all controls computed once at $t=0$:

| mode | $L_0$ | $\Phi(t)$ |
|---|---|---|
| translation | $0$ | $I$ |
| rotation, rigid | $A$ | $e^{tA}$ (in 2D, the rotation by the angle $t\theta$, see below) |
| scaling, translation+scaling | $B=\operatorname{diag}(b)$ | $\operatorname{diag}(e^{tb})$ |
| similarity with isotropic scaling | $A+bI$ | $e^{tb}\,e^{tA}$ |
| linear, affine | $L_0=S_0+K_0$ | $e^{2tK_0}\,e^{tL_0^\top}$ |
| rotation + anisotropic scaling | — | numerical integration |
| anything with non-rigid | — | numerical integration |

In 2D the rotation control and its flow are explicit:

$$
A=\begin{pmatrix}0&-\theta\\ \theta&0\end{pmatrix},\qquad
e^{tA}=\begin{pmatrix}\cos t\theta&-\sin t\theta\\ \sin t\theta&\cos t\theta\end{pmatrix},\qquad
\theta=\frac{1}{2c_A}\sum_i y_i\times p_{i,x},
$$

where $a\times b=a_1b_2-a_2b_1$ is the 2D cross product, so $\theta$ is the angular momentum about
the barycenter divided by $2c_A$. The 2D rigid map at $t=1$ is therefore a rotation by $\theta$ about
the barycenter, followed by a translation by $\tau=\tfrac1{c_T}\sum_ip_{i,x}$.
`Deformation.flow_map` returns it as $x\mapsto\Phi x+b$ with $b=c_0+\tau-\Phi c_0$.

The code is `linear_flow_trajectory` in `finitedim.py`. It uses `torch.linalg.matrix_exp`, so the
gradient with respect to the initial momenta comes from autograd. The cost is $O(Nd^2+d^3)$ per
requested time, against $O(n_t\,N d)$ autograd steps for the integrator. Backward shooting needs
nothing extra: $H$ is even in $p$, so running the closed form from the final state with both momenta negated,
$\big(q(1),w(1),-p_x(1),-p_w(1)\big)$, inverts the map
exactly.

### 4.7 Numerical checks

The test suite runs in float64 for the checks below.

- **Against a fine integrator.** For 14 mode, dimension and axis-group combinations, the closed form
  is compared with RK4 at $n_t=200$ and $400$. The relative error is below $10^{-7}$, and it drops
  16-fold each time $n_t$ doubles. That is RK4's fourth-order truncation error, so the closed form is
  the exact limit. A wrong formula would give errors around $10^{-2}$.
- **Conservation and inversion.** $H$ stays constant along closed-form trajectories to $10^{-10}$.
  Backward shooting returns $q$, $w$ and the passive points to $10^{-10}$.
- **Gradients.** Gradients through the closed form match gradients through RK4 for every entry of
  the final state, momenta included.
- **2D rigid map.** The 2D rigid flow map equals the rotation matrix of angle $A_{21}$ about the
  barycenter, plus $\tau$.

---

## 5. Running time

All timings come from `examples/rotation/benchmark_modes_yi.py`, run on 2026-11-03 on an Apple M4 Pro
CPU with 10 PyTorch threads, torch 2.11 and pykeops 2.3. No GPU was used. The raw numbers are in
`examples/rotation/output/benchmark/benchmark_results.json` and `.md`. To reproduce:

```
cd examples/rotation
KMP_DUPLICATE_LIB_OK=TRUE python benchmark_modes_yi.py --sizes 1000 10000 --repeats 3 \
    --npoints-registration 2000 --steps 10
```

The settings are those of the example script: $\sigma_{\mathrm{RKHS}} = (0.2, 0.1, 0.05)$,
$\sigma_{\mathrm{var}} = (0.2, 0.1, 0.05, 0.02)$, $c_A=c_T=c_{\mathrm{scale}}=c_L=1$, $c_{NR}=10$ and
$\gamma=0.1$. Numerical modes use the default Ralston scheme with $n_t=10$.

### 5.1 Deformation only

Each entry is the time of one deformation step as it happens inside a loss evaluation:
$\gamma H$, shooting to $t=1$, and the backward pass to the momenta. The points are random subsets
of the amygdala source with random momenta of moderate size. Each time is the median of 3 runs at
$N=1000$ and a single run at $N=10000$, after one warm-up call.

**Closed form against the same mode integrated numerically.** The "Ralston" columns force
numerical integration of the same mode. The last column is the error of that integration at
$N=10000$, relative to the largest displacement.

| mode | closed form, N=1000 | Ralston, N=1000 | speed-up | closed form, N=10000 | Ralston, N=10000 | speed-up | Ralston error |
|---|---|---|---|---|---|---|---|
| translation | 0.21 ms | 1.77 ms | 8× | 0.61 ms | 3.23 ms | 5× | 2.0e-07 |
| rotation | 0.39 ms | 6.88 ms | 18× | 0.87 ms | 15.28 ms | 18× | 2.4e-05 |
| scaling | 0.45 ms | 7.30 ms | 16× | 0.98 ms | 17.92 ms | 18× | 1.3e-04 |
| rigid | 0.44 ms | 8.14 ms | 18× | 0.94 ms | 15.82 ms | 17× | 1.2e-06 |
| similarity | 0.64 ms | 14.78 ms | 23× | 1.15 ms | 28.81 ms | 25× | 1.2e-06 |
| affine | 0.66 ms | 7.53 ms | 11× | 1.09 ms | 16.16 ms | 15× | 6.4e-06 |

**Numerically integrated modes**, with the original `xmodmap` as reference:

| mode | forward, N=1000 | fwd+bwd, N=1000 | forward, N=10000 | fwd+bwd, N=10000 |
|---|---|---|---|---|
| all-mixed (xmodmap) | 0.158 s | 0.999 s | 15.1 s | 89.4 s |
| all-mixed | 0.162 s | 0.939 s | 14.8 s | 94.5 s |
| non-rigid | 0.157 s | 0.901 s | 15.3 s | 92.9 s |
| non-rigid+rigid | 0.156 s | 0.919 s | 14.9 s | 96.1 s |

What the tables show:

- **The closed form is 15 to 25 times faster** than integrating the same mode, except for translation,
  where integration is already trivial. It also removes the time-discretization error in the last
  column. That error grows with the size of the motion, and these test momenta are moderate.
- **Finite-dimensional modes are essentially free.** At $N=10000$ one closed-form step takes about
  1 ms. One non-rigid step takes about 93 s, roughly $10^5$ times more.
- **Non-rigid cost grows quadratically.** It rises about 100-fold from $N=1000$ to $N=10000$, as
  expected for the $N^2$ kernel sums. The closed-form cost roughly doubles.
- **The refactor adds no overhead.** `all-mixed` in `xmodmap2` runs within ±6% of `xmodmap`, faster
  at $N=1000$ and slower at $N=10000$. That is within the run-to-run spread of single runs. Adding
  rigid modes to the non-rigid one costs almost nothing.

### 5.2 Full registration on the rotation example

The setup is that of `framework_script_2d_rotation_yi.py` on a random subset of $N=2000$ source
points. The target is the source rotated by 0.6 rad and translated, and the optimizer runs at most
10 steps. Accuracy is measured in rescaled units, where the shape fits in the unit box. The angle
and translation come from the closed-form map, or from a weighted Procrustes fit for the numerical
modes. The RMS is computed with the known point correspondences.

| mode | flow | s per loss eval. | optimizer time | loss evals | recovered angle | angle error | translation error | RMS to target | final data loss |
|---|---|---|---|---|---|---|---|---|---|
| translation | closed form | 0.043 | 0.4 s | 11 | 0.0000 | 6.0e-01 | 3.8e-01 | 1.8e-01 | 1.7e+01 |
| rotation | closed form | 0.035 | 0.5 s | 14 | -0.4456 | 1.0e+00 | 4.0e-01 | 4.3e-01 | 4.7e+01 |
| scaling | closed form | 0.036 | 0.3 s | 11 | 0.0000 | 6.0e-01 | 6.3e-01 | 4.1e-01 | 3.7e+01 |
| rigid | closed form | 0.043 | 1.8 s | 36 | 0.5998 | 1.8e-04 | 1.1e-04 | 5.3e-05 | 2.3e-05 |
| similarity | closed form | 0.048 | 1.7 s | 40 | 0.5998 | 1.8e-04 | 1.1e-04 | 5.2e-05 | 7.6e-06 |
| affine | closed form | 0.044 | 2.8 s | 57 | 0.5999 | 1.4e-04 | 9.5e-05 | 5.5e-05 | 1.5e-05 |
| non-rigid | Ralston nt=10 | 3.694 | 401.2 s | 114 | -0.1308 | 7.3e-01 | 4.2e-01 | 2.7e-01 | 1.3e+00 |
| non-rigid+rigid | Ralston nt=10 | 3.464 | 403.7 s | 116 | 0.5997 | 2.9e-04 | 1.1e-04 | 1.4e-03 | 5.7e-03 |
| all-mixed | Ralston nt=10 | 3.597 | 406.8 s | 116 | 0.5998 | 1.8e-04 | 1.6e-04 | 1.2e-03 | 4.1e-03 |

What the table shows:

- **Rigid, similarity and affine solve the problem in 2 to 3 seconds.** They recover the 0.6 rad
  rotation to about $2\times10^{-4}$ rad and the translation to about $10^{-4}$. The remaining
  angle bias comes from the regularization $\gamma H$, which slightly penalizes large rotations.
- **All-mixed reaches the same angle but is about 230 times slower.** It takes 407 s against 1.8 s.
  The non-rigid mode also absorbs a small part of the motion, so its RMS to the target is about 20
  times larger than the rigid mode's.
- **Loss evaluations dominate the cost.** At $N=2000$ a closed-form loss evaluation takes about
  0.04 s and is almost entirely the varifold term. A numerical evaluation takes about 3.6 s, so it
  costs about 84 times more.
- **Non-rigid alone fails within 10 steps.** It ends far from the target, at angle −0.13 with a
  data loss of 1.3. A 0.6 rad rotation is expensive to build from kernel velocities at
  $c_{NR}=10$ and $\gamma=0.1$.
- **Single-component modes cannot represent the target, as expected.** Translation alone, rotation
  alone and scaling alone need a combination of motions they do not have. Rotation alone also lands
  in a wrong local minimum at −0.45 rad.

Two caveats apply. These are single runs on one CPU. At $N=10000$ each Part A entry is a single
measurement, so differences under about 10% are not significant.

---

## 6. Tests

```
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_xmodmap2_deformation.py -q
55 passed, 1 xfailed
```

| test | checks |
|---|---|
| `test_parse_mode*`, `test_cS_is_rejected_with_explanation`, `test_nonrigid_needs_sigma` | mode strings, aliases, errors |
| `test_all_mixed_matches_xmodmap` (2D; 3D xy+z; 3D single) | H, shot states and gradients equal to `xmodmap`. The 3D case with an effective dimension of 2 is an expected failure because `xmodmap` crashes there |
| `test_grid_and_backward_shooting_match_xmodmap` | passive points equal to the former `ShootingGrid` and `ShootingBackwards` |
| `test_closed_form_matches_rk4` (14 cases) | exact flow is the limit of RK4, with fourth-order convergence |
| `test_closed_form_conserves_H_and_is_invertible` (14 cases) | conservation of $H$, exact backward inversion |
| `test_closed_form_gradient_matches_numeric` | autograd through `matrix_exp` |
| `test_rotation_plus_anisotropic_scaling_is_integrated_numerically` | non-constant controls in the non-solvable case |
| `test_2d_rigid_flow_map_is_rotation_about_barycenter` | explicit 2D formula |
| `test_common_rescaling_of_coefficients_equals_rescaling_gamma`, `test_cNR_is_an_explicit_inverse_gain` | the coefficient identities of section 3 |
| `test_deformation_api`, `test_single_modality_model_runs` | API, energies, weight, model and optimizer loop |

The environment is the `myenv` conda environment, with torch 2.11 and pykeops 2.3. Set
`KMP_DUPLICATE_LIB_OK=TRUE` because two OpenMP runtimes are linked.

---

## 7. Limitations and possible next steps

- **Overparametrized momenta in rigid modes.** These modes still optimize the full momenta, with
  $N\times d$ and $N$ entries, although the map depends only on $(L_0,\tau)$: 3 numbers for 2D
  rigid. LBFGS copes, since the gradient lies in that small subspace. Optimizing the controls
  directly would be cheaper and better conditioned.
- **Sequential composition.** "+" sums modes with a shared momentum, as `xmodmap` did. A sequential
  composition, a rigid map followed by a non-rigid one with separate momenta, would decouple
  the modes completely. It would also let the rigid stage use the closed form inside a non-rigid
  registration. That requires a model with two sets of momenta.
- **In-plane rotation for 3D data with an effective dimension of 2.** Restricting the rotation to
  the xy plane would make `similarity` closed-form there too. The current code keeps the full
  3D rotation of `xmodmap`.
- **Laplace kernel.** It needs a smooth regularization before it can be used with mass momenta. See
  section 2.6.
