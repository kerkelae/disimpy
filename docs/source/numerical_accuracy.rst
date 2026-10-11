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
