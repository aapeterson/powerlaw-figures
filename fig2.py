#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot
from scipy.integrate import solve_ivp
from scipy.optimize import fmin, fminbound
from scipy.special import gamma as gamma_fxn


def makefig(figsize=(5., 5.), nrows=1, ncols=1,
            lm=0.1, rm=0.1, bm=0.1, tm=0.1, hg=0.1, vg=0.1,
            hr=None, vr=None):
    """
    figsize: canvas size
    nrows, ncols: number of horizontal and vertical images
    lm, rm, bm, tm, hg, vg: margins and "gaps"
    hr is horizontal ratio, and can be fed in as for example
    (1., 2.) to make the second axis twice as wide as the first.
    [same for vr]
    """
    nv, nh = nrows, ncols
    hr = np.array(hr, dtype=float) / np.sum(hr) if hr else np.ones(nh) / nh
    vr = np.array(vr, dtype=float) / np.sum(vr) if vr else np.ones(nv) / nv

    axwidths = (1. - lm - rm - (nh - 1.) * hg) * hr
    axheights = (1. - bm - tm - (nv - 1.) * vg) * np.array(vr)

    fig = pyplot.figure(figsize=figsize)
    axes = []
    bottompoint = 1. - tm
    for iv, v in enumerate(range(nv)):
        leftpoint = lm
        bottompoint -= axheights[iv]
        for ih, h in enumerate(range(nh)):
            ax = fig.add_axes((leftpoint, bottompoint,
                               axwidths[ih], axheights[iv]))
            axes.append(ax)
            leftpoint += hg + axwidths[ih]
        bottompoint -= vg
    return fig, axes


def add_inset(fig, axes):
    """Adds the inset fig to hold the distro plot."""
    axes += [fig.add_axes((0.35, 0.55, 0.2, 0.4))]


class GammaDist:
    """Just a tool to make/plot the gamma distribution."""

    def __init__(self, k, mean=1.):
        self.k = k
        self.mean = mean

    def __call__(self, x):
        k, mean = self.k, self.mean
        y = x**(k - 1.)
        y *= np.exp(- x * k / mean)
        y /= (mean / k)**k
        y /= gamma_fxn(k)
        return y


class Get_dydt:
    """Creates the callable function needed by scipy's integrator.
    This exists as a class to make it easier to set parameters.
    """

    def __init__(self, beta, gamma, power):
        self.beta = beta  # infection rate, d^-1
        self.gamma = gamma  # destruction rate, d^-1
        self.power = power  # order in S

    def __call__(self, t, y):
        I, R = y
        S = 1. - I - R
        dI_dt = (self.beta * I * S**self.power - self.gamma * I)
        dR_dt = self.gamma * I
        return [dI_dt, dR_dt]


def load_Rs():
    with open('H1N1-R.csv', 'r') as f:
        lines = f.read().splitlines()[3:]
    Rs = []
    for line in lines:
        R = line.split('\t')[4]
        if '–' in R:
            # Toss out any containing ranges since they don't have a mean.
            continue
        Rs += [float(R)]
    return np.array(Rs)


def load_ever_infecteds():
    with open('H1N1.csv', 'r') as f:
        lines = f.read().splitlines()[1:]
    ever_infecteds = []
    for line in lines:
        ever_infecteds += [float(line.split('\t')[-1])]
    return np.array(ever_infecteds)


def common_integrations(R0, I_0=0.0001):
    """Performs common integrations for several panels."""
    # Exponential (zeroth-order) model.
    times = np.linspace(0., 33.)
    exp = {'I': np.array([I_0 * np.exp((R0 - 1.) * t) for t in times]),
           'R': np.array([I_0 / (R0 - 1.) * (np.exp((R0 - 1.) * t) - 1.)
                          for t in times])}
    # First-order.
    get_dydt = Get_dydt(beta=R0, gamma=1., power=1.)
    ans = solve_ivp(fun=get_dydt, y0=[I_0, 0.],
                    t_span=(times[0], times[-1]), dense_output=True)
    I, R = ans.sol(times)
    first = {'I': I, 'R': R}
    # Second-order.
    get_dydt = Get_dydt(beta=R0, gamma=1., power=2.)
    ans = solve_ivp(fun=get_dydt, y0=[I_0, 0.],
                    t_span=(times[0], times[-1]), dense_output=True)
    I, R = ans.sol(times)
    second = {'I': I, 'R': R}
    # 80:20.
    get_dydt = Get_dydt(beta=R0, gamma=1., power=1. + 1. / 0.245)
    ans = solve_ivp(fun=get_dydt, y0=[I_0, 0.],
                    t_span=(times[0], times[-1]), dense_output=True)
    I, R = ans.sol(times)
    fifth = {'I': I, 'R': R}
    return times, exp, first, second, fifth


class GetLoss:
    """Used to find the solution to the implicit final-size equations."""
    def __init__(self, Rsub0, I0=0., S0=1., power=1.):
        self.Rsub0 = Rsub0
        self.I0 = I0
        self.S0 = S0
        self.power = power

    def __call__(self, Z):
        if self.power == 1:
            lhs = Z
            rhs = self.S0 * (1. - np.exp(-self.Rsub0 * (Z + self.I0)))
            return (lhs - rhs)**2
        else:
            lhs = Z * (1. - self.power) * self.Rsub0
            rhs = 1. - 1. / (1. - Z)**(self.power - 1.)
            return (lhs - rhs)**2


def get_final_size(Rsub0, order=1):
    """Returns the final size of the epidemic for different orders in S."""
    if Rsub0 < 1.:
        return 0.
    if order == 1:
        getloss = GetLoss(Rsub0=Rsub0)
        if Rsub0 < 1.05:
            guess = 0.04
        elif Rsub0 < 1.1:
            guess = 0.11
        else:
            guess = 0.4
        return fmin(func=getloss, x0=guess)[0]
    elif order == 2:
        S_inf = 1. / Rsub0
        Z = 1. - S_inf
        return Z
    else:
        getloss = GetLoss(Rsub0=Rsub0, power=order)
        guess = 0.1
        return fminbound(func=getloss, x1=0.03, x2=1.0)


def panel_b(times, exp, first, second, fifth):
    """Log scale of initial growth."""
    ax = axes[0]
    ax.semilogy(times, 1. * exp['I'], '-', color='0.8', lw=5.,
                label="0th-order", zorder=1)
    ax.semilogy(times, 1. * first['I'], label='1st-order')
    ax.semilogy(times, 1. * second['I'], label='2nd-order')
    ax.semilogy(times, 1. * fifth['I'], label='5th-order')
    ax.set_xlim(0., 15.)
    ax.set_ylim(1e-5, 1e-1)
    ax.set_ylabel('active cases ($I$)')
    ax.set_xlabel('time (dimensionless)')
    ax.legend()


def panel_c(times, first, second, fifth, exp):
    ax = axes[1]
    ax.plot(times, 100. * (first['I'] + first['R']), label='1st-order')
    ax.plot(times, 100. * (second['I'] + second['R']), label='2nd-order')
    ax.plot(times, 100. * (fifth['I'] + fifth['R']), label='5th-order')
    ax.plot(times, 100. * (exp['I'] + exp['R']), color='0.8', lw=5.,
            zorder=1)
    ax.set_ylim(0., 100.)
    ax.set_xlim(0., times[-1])
    ax.set_xlim(left=0.)
    ax.set_ylabel('total infected (%)')
    ax.set_xlabel('time (dimensionless)')
    ax.text(times[-1] - 1., 100. * (first['I'] + first['R'])[-1] + 3.,
            'homogeneous (1st)', ha='right')
    ax.text(times[-1] - 1., 100. * (second['I'] + second['R'])[-1] + 3.,
            'exponential (2nd)', ha='right')
    ax.text(times[-1] - 1., 100. * (fifth['I'] + fifth['R'])[-1] + 3.,
            '"80:20 rule" (5th)', ha='right')


def panel_d():
    ax = axes[2]
    ever_infecteds = load_ever_infecteds()
    infected1s = [100. * get_final_size(Rsub0=_, order=1) for _ in Rs]
    infected2s = [100. * get_final_size(Rsub0=_, order=2) for _ in Rs]
    infected5s = [100. * get_final_size(Rsub0=_, order=5) for _ in Rs]
    for R, i1, i2 in zip(Rs, infected1s, infected2s):
        print('{:5.2f} {:6.2f} {:6.2f}'.format(R, i1, i2))
    boxes = ax.boxplot([infected1s, infected2s, infected5s, ever_infecteds],
                       vert=True,
                       whis=(5., 95.),
                       sym='',
                       patch_artist=True)
    boxes['boxes'][1].set_facecolor('C1')
    boxes['boxes'][2].set_facecolor('C2')
    boxes['boxes'][3].set_facecolor('0.5')
    for line in boxes['medians']:
        line.set_color('k')
    ax.plot([3.5] * 2, [0., 100.], ':k')
    ax.set_ylim(0., 100.)
    ax.set_ylabel('final size of 2009 H1N1 outbreak (%)')
    ax.set_xticklabels(['1st', '2nd', '5th', 'serum'])
    ax.tick_params(axis='x', which=u'both', length=0)


def inset_gamma_panel(ax):
    xs = np.linspace(0., 3., num=100)
    for k, color in zip([0.25, 1.], ['C2', 'C1']):
        gammas = [GammaDist(k=k)(_) for _ in xs]
        lw = 2. if k == 1 else 1.
        ax.plot(xs, gammas, label='$k{{=}}{:g}$'.format(k), lw=lw, color=color)
        ax.fill_between(xs, gammas, color=color, alpha=0.5)
    # k = infinity is a dirac-delta.
    ax.set_autoscale_on(False)
    ax.plot([0., xs[-1]], [0., 0.], color='C0', label=r'$k{\to}\infty$')
    ax.plot([1., 1.], [0., 100.], color='C0')
    ax.legend()
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('population density')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_letters():
    ax = axes[0]
    position = ax.get_position()
    fig.text(position.x0 + 0.015, position.y1 - 0.05, 'A',
             weight='bold', size=12., ha='center', va='center')
    ax = axes[1]
    position = ax.get_position()
    fig.text(position.x0 + 0.015, position.y1 - 0.05, 'B',
             weight='bold', size=12., ha='center', va='center')
    ax = axes[2]
    position = ax.get_position()
    fig.text(position.x0 + 0.015, position.y1 - 0.05, 'C',
             weight='bold', size=12., ha='center', va='center')


fig, axes = makefig(figsize=(10., 3.5), ncols=3, hr=(1., 3., 1.2), nrows=1,
                    rm=0.01, lm=0.07, bm=0.13, tm=0.025, hg=0.08)
add_inset(fig, axes)
Rs = load_Rs()
R0 = np.median(Rs)
times, exp, first, second, fifth = common_integrations(R0=R0)
panel_b(times, exp, first, second, fifth)
panel_c(times, first, second, fifth, exp)
panel_d()
inset_gamma_panel(axes[-1])
add_letters()
fig.savefig('fig2.pdf')
