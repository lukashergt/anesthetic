"""Tension statistics between two or more datasets."""
import numpy as np
from scipy.stats import chi2
from scipy.special import erfcinv, comb
from anesthetic.samples import Samples


def _validate_levels(levels, columns):
    n = len(levels)
    if n < 2:
        raise ValueError("Need at least two levels, single-dataset stats "
                         "`d_A` and `d_B` and the full joint stats `d_AB`.")
    nsamples = len(levels[-1][0])
    for k, level in enumerate(levels, start=1):
        expected = comb(n, k, exact=True)
        if len(level) != expected:
            raise ValueError(f"Level k={k} should have {expected} entries "
                             f"(C({n},{k})), got {len(level)}.")
        for i, stats_table in enumerate(level, start=1):
            if len(stats_table) != nsamples:
                raise ValueError(f"The stats tables need to have equal size "
                                 f"{nsamples}, got level k={k}, table number "
                                 f"{i} with {len(stats_table)} entries.")
            if not set(columns).issubset(stats_table.drop_labels().columns):
                raise ValueError(f"All provided stats objects must contain "
                                 f"the same columns: {columns}.")
    return n, nsamples


def _level_sum(level, column):
    out = level[0][column].copy()
    for stat_table in level[1:]:
        out += stat_table[column]
    return out


def _outer_sum(levels, column):
    return _level_sum(levels[0], column) - levels[-1][0][column]


def _alternating_sum(levels, column):
    """Alternating-sign sum over subset sizes for `column`."""
    out = _level_sum(levels[0], column)
    for k, level in enumerate(levels[1:], start=2):
        sign = +1 if (k % 2 == 1) else -1
        out += sign * _level_sum(level, column)
    return out


def tension_stats(*levels):
    r"""Compute tension statistics between two or more samples.

    With the Bayesian (log-)evidence ``logZ``, Kullback--Leibler divergence
    ``D_KL``, posterior average of the log-likelihood ``logL_P``, Gaussian
    model dimensionality ``d_G``, we can compute tension statistics between
    two or more samples (example here for simplicity just with two datasets
    A and B):

    - ``logR``: R statistic for dataset consistency.

      .. math::
        \ln R = \ln Z_{AB} - \ln Z_{A} - \ln Z_{B}

    - ``I``: Mutual information estimate between data and params:
      :math:`I(\Theta,A,B)`.

      .. math::
        \hat{I} = D_{KL}^{A} + D_{KL}^{B} - D_{KL}^{AB}

    - ``logS``: Suspiciousness.

      .. math::
        \ln S = \ln L_{AB} - \ln L_{A} - \ln L_{B}

    - ``d_G``: Gaussian model dimensionality of shared constrained parameters.

      .. math::
        d = d_{A} + d_{B} - d_{AB}

    - ``p``: p-value for the tension between two samples based on `logS`.

      .. math::
        p = \int_{d-2\ln{S}}^{\infty} \chi^2_d(x) dx

    - ``sigma``: Tension quantification in terms of numbers of sigma
      calculated from `p`.

      .. math::
        \sqrt{2} {\rm erfc}^{-1}(p)

    Parameters
    ----------
    *levels : tuple of :class:`anesthetic.samples.Samples`
        Bayesian stats from MCMC or nested sampling runs grouped by subset
        size. Each element of ``levels`` is a tuple of ``stats`` objects with
        columns ['logZ', 'D_KL', 'logL_P', 'd_G'] as returned by
        :meth:`anesthetic.samples.NestedSamples.stats` (``logZ`` and ``D_KL``
        are optional; the function will compute only the corresponding
        tension statistics if present).

        The expected structure is::

            levels[0] : (stats_A, stats_B, ..., stats_n)          # size C(n,1)
            levels[1] : (stats_AB, stats_AC, ..., stats_{n-1,n})  # size C(n,2)
            ...
            levels[n-1] : (stats_{A...n},)                        # size C(n,n)

        Examples::

            tension_stats((stats_A, stats_B), (stats_AB,))
            tension_stats((stats_A, stats_B, stats_C),
                          (stats_AB, stats_AC, stats_BC),
                          (stats_ABC,))

    Returns
    -------
    samples : :class:`anesthetic.samples.Samples`
        DataFrame containing the following tension statistics in columns:
        ['logR', 'I', 'logS', 'd_G', 'p', 'sigma']

    samples : :class:`anesthetic.samples.Samples`
        DataFrame containing the following tension statistics in columns:
        ['logR', 'I', 'logS', 'd_G', 'p', 'sigma'] (as available). For
        ``n`` datasets, additional columns ['logR_n', 'I_n', 'logS_n'] are
        included when the corresponding inputs are available, giving the
        irreducible n-way interaction component, i.e. `logS` is the full
        "cumulative" statistic, whereas `logS_n` is the specific contribution
        from the n-th level.
    """
    # Determine which columns are available across all stats objects
    columns = ["logL_P", "d_G"]
    full_joint = levels[-1][0]
    if "logZ" in full_joint.drop_labels().columns:
        columns.append("logZ")
    if "D_KL" in full_joint.drop_labels().columns:
        columns.append("D_KL")

    # Validate and infer n
    n, nsamples = _validate_levels(levels, columns)

    # Use the index of the full-joint stats to define output index
    samples = Samples(index=full_joint.index.copy())

    if "logZ" in columns:
        # Total n-way evidence ratio R
        samples["logR"] = -_outer_sum(levels, "logZ")
        samples.set_label("logR", r"$\ln\mathcal{R}$")
        # irreducible n-way interaction component of R
        samples[f"logR_{n}"] = _alternating_sum(levels, "logZ")
        samples.set_label(f"logR_{n}", fr"$\ln\mathcal{{R}}_{{{n}}}$")

    if "D_KL" in columns:
        # Total n-way estimate of data-parameter interaction information
        samples["I"] = _outer_sum(levels, "D_KL")
        samples.set_label("I", r"$\hat{\mathcal{I}}$")
        # irreducible n-way interaction component of I
        samples[f"I_{n}"] = _alternating_sum(levels, "D_KL")
        samples.set_label(f"I_{n}", fr"$\hat{{\mathcal{{I}}}}_{{{n}}}$")

    # Total n-way likelihood ratio or suspicousness S
    samples["logS"] = -_outer_sum(levels, "logL_P")
    samples.set_label("logS", r"$\ln\mathcal{S}$")
    # irreducible n-way interaction component of S
    samples[f"logS_{n}"] = _alternating_sum(levels, "logL_P")
    samples.set_label(f"logS_{n}", fr"$\ln\mathcal{{S}}_{{{n}}}$")

    # Number of "shared-by-all" parameters at level n
    samples["d_G"] = _alternating_sum(levels, "d_G")
    samples.set_label("d_G", r"$d_\mathrm{G}$")

    # p-value using the standard chi-square mapping:
    # X = d - 2 ln S ~ chi^2_d
    X = samples["d_G"] - 2 * samples["logS"]
    p = chi2.sf(X, df=samples["d_G"])
    samples["p"] = p
    samples.set_label("p", "$p$")

    samples["sigma"] = erfcinv(p) * np.sqrt(2)
    samples.set_label("sigma", r"$\sigma$")

    return samples
