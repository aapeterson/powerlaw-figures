This repository contains the complete code necessary to re-create the figures in the power-law SIR manuscript available at:

https://arxiv.org/abs/2005.04704

In this work, we show that variability in individual susceptibility fundamentally alters the order of SIR and related epidemiological models. If the distribution of susceptibilities in a population can be described with a gamma distribution with shape parameter k, then the force of infection in SIR-like models reduces to a power-law model:

r_I = beta I S^p

where p is given rigorously by p = 1 + 1 / k. That is, the SIR model can be expressed as

dS/dt = - beta I S^p

dI/dt = + beta I S^p - gamma I

dR/dt = gamma I

The implementation of this model to create the three figures is contained in the three python scripts in this repository, which require numpy, scipy, and matplotlib to run. If you are looking for an implementation of the model, fig2.py has this for an SIR model, while fig3.py has this for an SEIR model.
