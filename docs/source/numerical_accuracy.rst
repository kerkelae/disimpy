*******************************
Numerical accuracy and exchange
*******************************

Disimpy uses isotropically oriented steps of fixed length

.. math::

   \ell = \sqrt{6D\,\Delta t}.

For free diffusion this gives the requested mean squared displacement after an
integer number of steps. A finite collection of fixed-length steps is not the
Gaussian Brownian propagator, and boundary interactions and phase accumulation
retain finite-step error. Refine the time step before interpreting a result.

Gradient arrays contain one sample at each time point. ``N`` samples define
``N - 1`` diffusion steps over ``(N - 1) * dt``. The simulator uses the mean
of each interval's endpoint gradients and accumulates phase at the post-step
position. ``gradients.calc_b`` integrates the piecewise-linear waveform; its
continuous value and finite-step signal attenuation need not agree on a coarse
time grid.

From v0.4, ``gradients.calc_b`` evaluates the exact integral of
``|q(T) - q(t)|^2`` for the piecewise-linear gradient on these ``N - 1``
intervals. In v0.3, it applied the trapezoidal rule to ``|q(t)|^2`` at the
sample times. Even a rephased waveform can therefore receive a different
b-value on a coarse grid: for samples ``[0, G, -G, 0]``, the old value is
``0.50 * (gamma * G)^2 * dt^3`` and the new value is ``0.55`` times that scale.
For a nonrephased two-sample waveform ``[G, G]``, the same change gives
``0.50`` versus ``1/3`` times that scale; subtracting ``q(T)`` changes the
convention as well as the integration rule.
If you migrate saved gradients that were scaled with v0.3 ``set_b``, recompute
their b-values with v0.4; use ``set_b`` again when the same target b-value is
needed. The waveform amplitudes can change as a result.

For a rephased free waveform, let ``q_j`` denote the q-vector at sample ``j``
and ``q_T`` its final value. The exact normalized signal for the discrete walk
is

.. math::

   S_{\mathrm{fixed}} = \prod_{j=0}^{N-2}
   \operatorname{sinc}\!\left(\ell\lvert q_T-q_j\rvert\right),

where ``sinc(x) = sin(x)/x``. Comparing this reference with a free simulation
separates Monte Carlo error from the step model's error.

``substrates.mesh(..., perm_prob=p)`` sets the probability of crossing a
membrane triangle on each encounter. The argument is dimensionless and tied
to the selected diffusivity and time step; holding ``p`` fixed while changing
``dt`` is not a constant physical permeability experiment. Artificial voxel
walls in nonperiodic meshes remain impermeable.

For restricted simulations, repeat the calculation with smaller ``dt`` while
holding the geometry, total diffusion time, and gradient waveform fixed.
Compare differences with Monte Carlo uncertainty from independent seeds.
Refine the mesh independently, vary its collision-grid subdivision ``n_sv``,
and check sensitivity to ``epsilon``. The step length must resolve relevant
gaps and curvature, and the collision limit ``max_iter`` must be high enough
to resolve repeated reflections. There is no geometry-independent safe ratio
of step length to feature size.
