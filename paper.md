---
title: 'Disimpy: A massively parallel Monte Carlo simulator for synthesizing diffusion-weighted MRI data in Python'
tags:
  - Python
  - diffusion MRI
  - neuroscience
  - microstructure
authors:
  - name: Leevi Kerkelä
    orcid: 0000-0001-9446-4305
    affiliation: 1
  - name: Fabio Nery
    affiliation: 1
  - name: Matt G. Hall
    affiliation: "2,1"
  - name: Chris A. Clark
    affiliation: 1
affiliations:
 - name: UCL Great Ormond Street Institute of Child Health, University College London, London, United Kingdom
   index: 1
 - name: National Physical Laboratory, Teddington, United Kingdom
   index: 2
date: 15 July 2020
bibliography: paper.bib

---

# Summary

Disimpy is a simulator for synthesizing diffusion-weighted magnetic resonance
imaging (dMRI) data that is useful in the development and validation of new
methods for data acquisition and analysis. Diffusion of water is modelled as an
ensemble of random walkers whose trajectories are generated on an Nvidia (Nvidia
Corporation, Santa Clara, California, United States)
CUDA-capable [@nickolls:2008]⁠ graphical processing unit (GPU). The massive
parallelization results in a significant performance gain, enabling simulation
experiments to be performed on standard laptop and desktop computers. Disimpy is
written in Python (Python Software Foundation), making its source code very
approachable and easily extensible.

# Statement of need

Since the diffusion of water in biological tissues is restricted by microscopic
obstacles such as cell organelles, myelin, and macromolecules, dMRI enables the
study of tissue microstructure *in vivo* by probing the displacements of water
molecules [@behrens:2009]⁠. It has become a standard tool in
neuroscience [@assaf:2019]⁠, and a large number of data acquisition and analysis
methods have been developed to tackle the difficult inverse problem of inferring
microstructural properties of tissue from the dMRI signal [@novikov:2019]⁠. 
Simulations have played an important role in the development of the field
because they do not require the use of expensive scanner time and they provide a
powerful tool for investigating the accuracy and precision of new methods, e.g.,
[@tournier:2007]. Here, we present a GPU-accelerated dMRI simulator that enables
a large amount of synthetic data to be generated on standard desktop and laptop
computers without needing to access high performance computing clusters.

# Features

Disimpy uses efficient numerical methods from Numpy [@walt:2011]⁠ and Scipy
[@virtanen:2020]⁠. Numba [@lam:2015]⁠ is used to compile Python code into
CUDA kernels and device functions [@nickolls:2008]⁠ which are executed on the
GPU. The random walker steps are generated in a massively parallel fashion on
individual threads of one-dimensional CUDA blocks, resulting in a performance
gain of over an order of magnitude when compared to Camino [@hall:2009]⁠⁠, a
popular dMRI simulator written in Java (\autoref{fig:1}). Given that random
walker Monte Carlo dMRI simulations require at least $10^4$ random walkers for
sufficient convergence [@hall:2009]⁠, it is important that Disimpy's runtime does
not linearly depend on the number of random walkers until the number of walkers
is in the thousands or tens of thousands, depending on the GPU.

Diffusion can be simulated without restrictions, inside analytically defined
geometries (cylinders, spheres, ellipsoids), and in arbitrary geometries defined
by triangular meshes [@panagiotaki:2010; @hall:2017] (\autoref{fig:2}).
Importantly, the random walk model of diffusion is able to capture
time-dependent diffusion.

Disimpy supports arbitrary diffusion encoding gradient sequences, such as those
used in conventional pulsed field gradient experiments [@stejskal:1965] as well
as the recently developed q-space trajectory encoding 
[@eriksson:2013; @sjolund:2015]⁠. Useful helper functions for generating and
manipulating gradient arrays are provided. Synthetic data from multiple gradient
encoding schemes can be generated from the same simulation.

Documentation, tutorial, and contributing guidelines are provided at 
https://disimpy.readthedocs.io/.

# Signal generation

Signal generation in Disimpy follows the framework established in [@hall:2009]⁠
which is briefly summarized here. \autoref{eq:1} and \autoref{eq:2} describe
theory. \autoref{eq:3}, \autoref{eq:4}, and \autoref{eq:5} describe the
numerical implementation in Disimpy.

In a dMRI experiment, the target nuclei are exposed to time-dependent magnetic
field gradients which render the signal sensitive to diffusion. During the
experiment, the spin of a nucleus experiences a path-dependent phase shift given
by
\begin{equation}\label{eq:1}
\phi(t) = \gamma \int_0^t \mathbf{B}_0 + \mathbf{G}(t')\cdot\mathbf{r}(t') dt' ,
\end{equation}
where $\gamma$ is the gyromagnetic ratio of the nucleus, $\mathbf{B}_0$ is the
static main magnetic field of the scanner, $\mathbf{G}(t)$ is the diffusion
encoding gradient, and $\mathbf{r}(t)$ is the location of the nucleus.
$\mathbf{G}$ and $\mathbf{B}_0$ change sign after the application of the
refocusing pulse.

An imaging voxel contains an ensemble of nuclei for which the total signal is
given by
\begin{equation}\label{eq:2}
S = S_0 \int_{-\infty}^\infty P(\phi) \exp \left( i \phi \right) d\phi ,
\end{equation}
where $P$ is the spin phase distribution and $S_0$ is the signal without
diffusion-weighting while keeping other imaging parameters unchanged.

In Disimpy, diffusion is modelled as a three-dimensional random walk over
discrete time. The steps, which are randomly sampled from a uniform
distribution over the surface of a sphere using the xoroshiro128+ pseudorandom
number generator [@blackman:2018], have a fixed length
\begin{equation}\label{eq:3}
l = \sqrt{6 \cdot D \cdot dt} ,
\end{equation}
where $D$ is the diffusion coefficient and $dt$ is the duration of a time step.
At every time point in the simulation, each walker accumulates phase given by
\begin{equation}\label{eq:4}
d \phi = \gamma \mathbf{G}(t) \cdot \mathbf{r} (t) dt .
\end{equation}
At the end of the simulated dynamics, the normalized diffusion-weighted signal
is calculated as the sum of signals from all random walkers
\begin{equation}\label{eq:5}
S = \sum_{j=1}^N \text{Re} \left( \exp \left( i \phi_j \right) \right) ,
\end{equation}
where $N$ is the number of random walkers.

The initial positions of the random walkers are drawn from a uniform
distribution across the diffusion environment. When a random walker collides
with a restricting barrier, it is elastically reflected off the collision point
in such a way that the random walker's total path length during $dt$ is equal to
$l$.

# Figures

![Performance comparison between Disimpy and Camino, a popular dMRI simulator that runs on the CPU. The comparison was performed on a desktop computer with an Intel Xeon E5-1620 v3 3.50 GHz x 8 CPU and an Nvidia Quadro K620 GPU. The simulations were performed using a mesh consisting of $10^4$ triangles, shown in \autoref{fig:2}.\label{fig:1}](paper_figure_1.png)

![Example of diffusion in an environment defined by a triangular mesh. (A) Example mesh of $10^4$ triangles defining the synthetic voxel consisting of 100 spheres with gamma distributed radii. Mesh kindly provided by Gyori [@gyori:2020]⁠. (B) Example trajectories of 100 random walkers whose initial positions were randomly positioned inside the spheres. Some spheres contain more than one walker. (C) Example trajectories of 100 random walkers outside the spheres.\label{fig:2}](paper_figure_2.png)


# Acknowledgements

This work was funded by the National Institute for Health Research Great Ormond
Street Hospital Biomedical Research Centre (NIHR GOSH BRC).

# References
