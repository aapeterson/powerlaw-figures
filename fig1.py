import numpy as np
from matplotlib import pyplot
from matplotlib.patches import FancyArrowPatch as Arrow
from scipy.integrate import solve_ivp


def susceptibility_panels(top_axis, bottom_axis):
    susceptibilities = np.linspace(0., 4.2)
    densities = np.exp(-susceptibilities)
    probabilities = susceptibilities * densities

    ax = top_axis
    ax.plot(susceptibilities, densities, lw=2.)
    ax.fill_between(susceptibilities, densities, color='0.5')
    ax.plot([1.]*2, [0., .5], 'r-')
    ax.text(1., 0.51, r'mean = $\bar{\varepsilon}$', color='r')
    ax.text(0.12, 0.25, r'$e^{-\varepsilon/\bar{\varepsilon}}$',
            ha='center', weight='bold', transform=ax.transAxes)

    ax.set_ylabel('population density')
    ax.set_xlim(0., susceptibilities[-1])
    ax.set_ylim(0., 1.)
    ax.set_yticks([])

    ax = bottom_axis
    ax.plot(susceptibilities, probabilities, lw=2.)
    ax.fill_between(susceptibilities, probabilities, color='0.5')
    ax.plot([2.]*2, [0., .3], 'r-')
    ax.text(2., 0.31, r'mean = 2$\bar{\varepsilon}$', color='r')
    ax.text(0.28, .35,
            r'$\varepsilon \cdot e^{-\varepsilon/\bar{\varepsilon}}$',
            ha='center', weight='bold', transform=ax.transAxes)

    ax.set_xlabel(r'infection susceptibility '
                  r'($\varepsilon/\bar{\varepsilon}$)')
    ax.set_ylabel('draw probability')
    ax.set_xlim(0., susceptibilities[-1])
    ax.set_ylim(0., .45)
    ax.set_yticks([])


def binned_odes_panel(ax):
    infectious0 = 0.01
    decay_time = 7.  # days
    doubling_time = 4.  # days
    times = np.linspace(0., 75., num=10000)
    gamma = 1. / decay_time
    beta = np.log(2.) / doubling_time + gamma
    # Second-order model.
    get_dydt = Get_dydt(beta=beta, gamma=gamma, power=2., bin_means=[1.])
    y0 = [infectious0, 1. - infectious0]
    second = solve_ivp(fun=get_dydt, y0=y0, t_span=(times[0], times[-1]),
                       dense_output=True)
    yt = second.sol(times)
    infectious2 = yt[0] * 100.
    susceptible2 = yt[1] * 100.
    recovered2 = 100. - infectious2 - susceptible2
    ax.plot(times, infectious2 + recovered2, color='0.5', lw=2.,
            label='2nd-order')
    # First-order model.
    get_dydt = Get_dydt(beta=beta, gamma=gamma, power=1., bin_means=[1.])
    y0 = [infectious0, 1. - infectious0]
    first = solve_ivp(fun=get_dydt, y0=y0, t_span=(times[0], times[-1]),
                      dense_output=True)
    yt = first.sol(times)
    infectious1 = yt[0] * 100.
    susceptible1 = yt[1] * 100.
    recovered1 = 100. - infectious1 - susceptible1
    ax.plot(times, infectious1 + recovered1, color='0.5', lw=2.,
            label='1st-order')
    # Shade in-between.
    ax.fill_between(times,
                    (infectious2 + recovered2), (infectious1 + recovered1),
                    color='0.8')
    ax.text(times[-1] - 2., (infectious1 + recovered1)[-1],
            '1st-order limit', ha='right', va='bottom')
    ax.text(times[-1] - 2., (infectious2 + recovered2)[-1] - 8.,
            '2nd-order limit', ha='right', va='top')
    # Binned 1st-order model.
    bin_choices = [1, 2, 3, 10]
    text_offsets = [0, +2, 0, -2]
    for n_bins, text_offset in zip(bin_choices, text_offsets):
        bins = Bins(initial_population=1. - infectious0, initial_mean=1.,
                    n_bins=n_bins)
        get_dydt = Get_dydt(beta=beta, gamma=gamma, power=1.,
                            bin_means=bins.bin_means)
        y0 = np.empty(len(bins) + 1)
        y0[0] = infectious0
        y0[1:] = bins.populations
        first = solve_ivp(fun=get_dydt, y0=y0, t_span=(times[0], times[-1]),
                          dense_output=True)
        yt = first.sol(times)
        infectious = yt[0] * 100.
        susceptibles = yt[1:] * 100.
        allsusceptibles = susceptibles.sum(axis=0)
        recovered = 100. - infectious - allsusceptibles
        ax.plot(times, infectious + recovered, '-k', lw=1.)
        label = '{:d} bins'.format(n_bins)
        if n_bins == 1:
            label = label[:-1]
        ax.text(times[-1] + 3., (infectious + recovered)[-1] + text_offset,
                label, ha='left', va='center', fontsize=7.)
    ax.set_xlim(0., times[-1] + 10.)
    ax.set_ylim(0., 100.)
    ax.set_xlabel('days')
    ax.set_ylabel('total infected (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


class Bins:
    """Data container to hold current state of bin data.
    The bins are created in a manner so that each has the same initial
    area; that is, the same population, when in an exponential distribution."""
    def __init__(self, initial_population, initial_mean, n_bins):
        N = initial_population
        kT = initial_mean
        self.n_bins = n_bins
        # The bins are defined by their left edges; the final bin has right
        # edge of +infinity.
        self.bin_left_edges = - kT * np.log(np.arange(1., 0., -1. / n_bins))
        # bin_means contains the mean value of energy within that bin; it is
        # not the true center due to the curvature of the exponential
        # distribution.
        self.bin_means = get_bin_means(self.bin_left_edges, kT)
        self.populations = (N / n_bins * np.ones_like(self.bin_left_edges))
        self.widths = self.bin_left_edges[1:] - self.bin_left_edges[:-1]

    def draw(self):
        """Randomly draws a sample from the population according to the
        susceptibility-weighted probability."""
        # For the logic of the draw, we make a long bar of which each bin
        # occupies a certain length. The left edge of this bar starts at 0,
        # prob_right_edges gives the right edge of each bin on this bar.
        prob_right_edges = np.cumsum(self.bin_means * self.populations)
        random = np.random.rand() * prob_right_edges[-1]
        chosen_bin = self.n_bins - sum(prob_right_edges > random)
        self.populations[chosen_bin] -= 1

    def get_mean(self):
        """Calculates the mean susceptibility score; that is, the
        instantaneous kT."""
        return (np.sum(self.populations * self.bin_means) /
                self.populations.sum())

    def __len__(self):
        return self.n_bins


def get_bin_means(bin_left_edges, kT):
    """Given a set of bins defined by their left edges (with the final
    right edge implicitly located at +infinity), calculate the mean value
    of an exponential distribution for each bin. Returns an array of the
    same length as the bin_left_edges array. kT is the temperature (in energy
    units) of the Boltzmann distribution. The output array will have the
    same units as kT."""
    means = np.empty_like(bin_left_edges)
    means[-1] = bin_left_edges[-1] + kT
    for index, bin_left_edge in enumerate(bin_left_edges[:-1]):
        left = bin_left_edge
        right = bin_left_edges[index + 1]
        expl = np.exp(-left / kT)
        expr = np.exp(-right / kT)
        means[index] = (((left + kT) * expl - (right + kT) * expr) /
                        (expl - expr))
    return means


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


class Get_dydt:
    """Creates the callable function needed by scipy's integrator.
    This exists as a class to make it easier to set parameters.
    """
    def __init__(self, beta, gamma, power, bin_means):
        self.beta = beta  # infection rate, d^-1
        self.gamma = gamma  # recovery rate, d^-1
        self.power = power  # order in S
        self.bin_means = bin_means  # array of means of the bins of suceptible

    def __call__(self, t, y):
        I = y[0]
        S = y[1:]
        dS_dt = (- self.beta * I * S**self.power * self.bin_means)
        dI_dt = - dS_dt.sum() - self.gamma * I
        dydt = np.empty_like(y)
        dydt[0] = dI_dt
        dydt[1:] = dS_dt
        return dydt


def convert_to_triangle(bins, integers=False):
    """Given a bins object, change to a triangle distribution (max at 0
    and 0 at max). If integers is True, will make all bins have integer
    populations (by simple rounding).
    """
    N = sum(bins.populations)
    n_bins = bins.n_bins
    initial_mean = bins.get_mean()
    right_edge = initial_mean * 3.
    bins.bin_left_edges = np.linspace(0., right_edge,
                                      num=n_bins, endpoint=False)
    bins.populations = np.linspace(2., 0., num=n_bins) * N / n_bins
    if integers:
        bins.populations = np.round(bins.populations)
    bins.widths = bins.bin_left_edges[1:] - bins.bin_left_edges[:-1]
    bins.bin_means = bins.bin_left_edges + bins.widths[0] / 2.


def make_evolution_panel(ax):
    N = 1e5  # population size
    n_bins = 100
    samples_to_plot = 5
    plot_frequency = int(N / samples_to_plot)
    bins = Bins(initial_population=N, initial_mean=1., n_bins=n_bins)
    convert_to_triangle(bins, integers=True)
    plot_normalizer = bins.populations[0]
    for _ in range(int(N)):
        if _ % plot_frequency == 0:
            print('plotting')
            print(_)
            population = bins.populations[:-1]
            p = ax.plot(bins.bin_means[:-1], population / plot_normalizer, '.')
            color = p[0].get_markerfacecolor()
            # Fit an exponential.
            mean = (np.sum(population * bins.bin_means[:-1]) /
                    np.sum(population))
            predicteds = [np.exp(-energy / mean) / mean
                          for energy in bins.bin_means[:-1]]
            predicteds = np.array(predicteds)
            predicteds *= np.sum(population) / np.sum(predicteds)
            ax.plot(bins.bin_means[:-1], predicteds / plot_normalizer, '-',
                    color=color)
        bins.draw()
    arrow = Arrow((2., 0.5), (.3, .1),
                  connectionstyle="arc3,rad=-.3", color='k', zorder=10,
                  arrowstyle='Simple,tail_width=0.5,head_width=4,'
                             'head_length=8')
    ax.add_patch(arrow)
    ax.text(2., .5, 'time')
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('population density')
    ax.set_ylim(0., 1.)
    ax.plot([], [], '.', color='0.5', label='Monte Carlo')
    ax.plot([], [], '-', color='0.5', label='exponential')
    ax.legend(frameon=False)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_abcd():
    ax = axes[0]
    position = ax.get_position()
    fig.text(position.x0 + 0.03, position.y1 - 0.03, 'A',
             weight='bold', size=12., ha='center', va='center')
    ax = axes[2]
    position = ax.get_position()
    fig.text(position.x0 + 0.03, position.y1 - 0.03, 'B',
             weight='bold', size=12., ha='center', va='center')
    ax = axes[1]
    position = ax.get_position()
    fig.text(position.x0 + 0.02, position.y1 - 0.02, 'C',
             weight='bold', size=12., ha='center', va='center')
    ax = axes[3]
    position = ax.get_position()
    fig.text(position.x0 + 0.10, position.y1 - 0.02, 'D',
             weight='bold', size=12., ha='center', va='center')


fig, axes = makefig(figsize=(7., 4.), nrows=2, ncols=2, hr=(1.3, 1.),
                    lm=0.03, tm=0.02, rm=0.05)
susceptibility_panels(axes[0], axes[2])
binned_odes_panel(axes[1])
make_evolution_panel(axes[3])
add_abcd()
fig.savefig('fig1.pdf')
