#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot
from scipy.integrate import solve_ivp
from scipy.optimize import fmin


class Make_dydt:
    """Makes the ODEs for the SEIR model.
    beta: S -> E
    nu: E -> I
    gamma: I -> R
    power: power on S.
    """
    def __init__(self, beta, nu, gamma, power=1.):
        self.beta = beta
        self.nu = nu
        self.gamma = gamma
        self.power = power

    def __call__(self, t, y):
        S, E, I = y
        r_E = self.beta * I * S**self.power
        r_I = self.nu * E
        r_R = self.gamma * I
        dydt = [- r_E,
                + r_E - r_I,
                + r_I - r_R]
        return dydt


def get_critical_time(solution, critical, guess=50.):
    """Given a solution object from the ODE solver, find the time at which
    the given critical concentration occurs. (This is taken to be cumulative
    infections.)"""
    def get_loss(time):
        susceptible = solution(time)[0] * 100.
        cumulative_infections = 100. - susceptible
        loss = (cumulative_infections - critical)**2
        return loss
    ans = fmin(func=get_loss, x0=guess)[0]
    return ans


def make_integration_panel(ax, R0):
    nu = 1. / 4.6
    gamma = 1. / 5.
    beta = gamma * R0
    E0 = 0.001
    I0 = 0.
    S0 = 1. - E0 - I0
    times = np.linspace(0., 150., num=200)
    powers = [(1., 'homogeneous'),
              (2., 'exponential'),
              (1. + 1./0.245, '80:20 rule')]
    for power, label in powers:
        dydt = Make_dydt(beta, nu, gamma, power)
        ans = solve_ivp(fun=dydt, t_span=times[[0, -1]], y0=[S0, E0, I0],
                        dense_output=True)
        Ss, Es, Is = ans.sol(times) * 100.
        line = ax.plot(times, 100. - Ss, label=label)
        pc = 100. * (1. - (gamma / beta)**(1./power))
        t = get_critical_time(ans.sol, pc, guess=50.)
        ax.plot([times[0], t], [pc] * 2, ':', lw=0.6,
                color=line[0].get_color())
        ax.text(times[-1] - 3., (100. - Ss)[-1] + 2., label, ha='right')
    ax.set_xlabel('days')
    ax.set_ylim(0., 100.)
    ax.set_xlim(times[[0, -1]])
    ax.text(0.18, 0.95, r'$\mathcal{{R}}_0{{=}}{:g}$'.format(R0),
            ha='center', va='center', transform=ax.transAxes)


fig, axes = pyplot.subplots(figsize=(8., 4.), ncols=2, sharex=True,
                            sharey=True)
fig.subplots_adjust(top=0.98, right=0.97, wspace=0.1, left=0.07)
make_integration_panel(axes[0], R0=2.2)
make_integration_panel(axes[1], R0=2.6)
axes[0].set_ylabel('total infected (%)')
fig.savefig('fig3.pdf')
