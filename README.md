# Project of simulating molecular dynamics (2015)

## Overview

One way of obtaining the macroscopic properties of matter is simulating its molecular dynamics [MD]. The underlying assumption here is that the values of specific parameters describing the system could be expressed as functions of microscopic parameters averaged over sufficiently long times (e.g. the temperature is related to average kinetic energy of particles involved). According to ergodic hypothesis this approach is equivalent to statistical physics approach based on statistical ensembles. However, simulating MD allows to avoid some cumbersome analytical calculations that may be involved in statistical physics approach.

Due to the sheer number of particles in a physically meaningful sample, simulating MD requires long calculation times - unless the numerics could be paralelised. The purpose of this project was to compare the times of simulating about 500 particles on CPU and on GPU, using NVidia graphic cards and CUDA programming tools.

This repository contains source codes of programs written for the project. More details are provided further below. Full overview together with the obtained results are provided in the original report (pdf file, currently only in Polish). This project uses some open source code attached to the book "CUDA by examples" by Jason Sanders and Edward Kandrot, hence the licence. Full list of used literature is provided at the end.

## Installation (on Linux systems)

Obviously, NVidia card with computing capabilities and nvcc compiler are required. Download all files to a chosen directory. In a terminal, go to the directory and run

`make`

to trigger Makefile. This will run commands to compile 5 programs, as well as create a directory "init_sets" with one file that may be used to initialise calculations. To remove the created files and the directory, type

`make clean`

## Synopsis of programs and scripts

**randv** *name* *N*

Creates a set of random initial velocities for particles obeying the rule that the centre of mass is at rest. These sets are named 'name.dat'. The particles are assumed to be initially located on a 3D grid $N \times N \times N$, and so each set contains data for $N^3$ particles.

**randoms** *name* *N*

This is a bash script. Creates five sets of random initial velocities like the above. These sets are named 'namei.dat' and saved in 'init_sets' folder. The particles are assumed to be initially located on a 3D grid $N \times N \times N$, and so each set contains data for $N^3$ particles.

**cpu_velocityverlet** *initial_velocities* *output_file* *b* *step* *number_of_steps*

Most basic program simulating the MD on the CPU. During calculations, it checks all the pairs of $N$ particles to take interactions into account, so the calculation time grows as $N^2$. The arguments are:

- initial_velocities: the name of the file. First line contains the length of the edge of simulated space (given in the number of particles alongside the edge). Subsequent lines should contain xyz coordinates of velocities of subsequent particles. Note that for physical reasons the average velocity should be zero, i.e. the center of mass should be at rest. Files in this format are created by randv described above.
- output_file: the name of the output file. First line have the information about parameters and calculation time. Next lines contain the values of kinetic energy, potential energy and total energy of the whole system.
- b - initial separation between particles. Basically, $b^{-3}$ is the density of particles. The values used in the report were from the range 1.0-4.0.
- step - the size of the time step. The value $h=0.01$ was used in the report. Note that the longer the time step, the less precise the calculations. If the total energy is not conserved over time, then the time step should be shorter.
- number_of_steps - how many steps would the simulation perform. The report indicates that, given the values above, about 100-200 steps are required to pass from the highly ordered initial state to a physically meaningful stable one.

**cpuopt_velocityverlet** *same arguments as above*

Calculations made by this program are in principle less accurate, but the time scales approximately like $N$ due to the use of so-called Verlet lists described further below.

**gpu_velocityverlet** and **gpuopt_velocityverlet** (*same arguments as above*)

Both use GPU lock-free synchronisation. The returned time of calculations does not take into account the time needed to initialise kernel and to copy information back to CPU.

**visualisation.py**

Simple *ad hoc* python script to visualise the data. It'll ask you for the name of the input and the output, where it plots kinetic, potential and total energy versus time.

## More details about the simulation

The interaction of any two particles in this simulation separated by $r$ is described by Lennard-Jones potential:

$$
V_{LJ} = 4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right] + const
$$

where $\epsilon $ is the depth of the density well and $\sigma $ is the value of $r$ for which the potential starts to be very small. Due to that, a cutoff could be introduced: the above equation is used only for $r<R_C$ and the potential is taken to be 0 otherwise. In order for the potential to be continuous at $r=R_C$, the constant is taken to be equal

$$
const = -4\epsilon\left[\left(\frac{\sigma}{R_C}\right)^{12} - \left(\frac{\sigma}{R_C}\right)^{6}\right]
$$

In the simulation, $R_C=2.5\sigma $. Both $\sigma $ and the mass of the particle are used as units in the simulation and hence are equal 1 by definition.

In principle to simulate MD one has to check the mutual interaction of every pair of particles. But, given the cutoff, some optimalisation may be introduced. For a given particle, one could assign a list - so called Verlet list - of particles that have the chance to interact in the near future. This list is actualised after a number of steps, by finding all the particles that are closer than certain treshold $R_M$. This number of steps is ussually given as a constant of simulation (dependent on other parameters). Here, a more "paranoic" version was applied: this number is calculated based on the largest velocity $v_{max}$ registered since the last actualisation: $R_M - R_C \simeq i\cdot 2v_{max} \cdot h$, where $i$ is the number of steps until next actualisation and $h$ is the timestep of the simulation. $R_M$ was taken to be equal $3.3\sigma$.

Initially, particles are located on a cubic grid with random velocities. In each step of a simulation, new positions and velocities are calculated by the Verlet algorithm (that approximates the proper Newton dynamics):

$$
\vec{r}(t+h) = \vec{r}(t) + \vec{v}(t)h + \frac{1}{2}\vec{a}(t)h^2
$$

$$
\vec{v}(t+h) = \vec{v}(t) + \frac{1}{2}[\vec{a}(t) + \vec{a}(t+h)]h
$$

## Literature

- J. Sanders, E. Kandrot, *CUDA by Example. An introduction to general-purpose GPU programming*, Addison-Wesley 2011, with examples and codes aviable at: przykłady do książki dostępne na: 
- M. P. Allen, *Introduction to Molecular Dynamics Simulation*, chapter 3
- L. Verlet, Phys. Rev. {\bf 159}(1), 98-103, 5 July 1967
- S. Xiao, W. Feng, *Inter-Block GPU Communication via Fast Barrier Synchronization*
- T. Lipscomb, A. Zou, S. S. Cho, *Parallel Verlet Neighbor List Algorithm for GPU-Optimized MD Simulations*, aviable at: http://users.wfu.edu/choss/docs/papers/21.pdf
