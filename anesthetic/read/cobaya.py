"""Read MCMCSamples from Cobaya chains."""
import os
import re
import numpy as np
from anesthetic.samples import MCMCSamples


def _count_samples(filename):
    """Count samples in a Cobaya chain file."""
    with open(filename) as f:
        return sum(bool(line.strip()) and not line.lstrip().startswith('#')
                   for line in f)


def read_paramnames(root):
    """Read header of ``<root>.1.txt`` to infer the paramnames.

    This is the data file of the first chain. It should have as many
    columns as there are parameters (sampled and derived) plus an
    additional two corresponding to the weights (first column) and the
    log-posterior (second column). The first line should start with a # and
    should list the parameter names corresponding to the columns. These
    will be used as handles in the pandas array.
    """
    with open(root + ".1.txt") as f:
        header = f.readline()[1:]
        paramnames = header.split()[2:]
        try:
            from getdist.cobaya_interface import cobaya_params_file
            from getdist.paramnames import ParamNames
            params = ParamNames(cobaya_params_file(root))
            labels = {p.name: '$' + p.label + '$' for p in params.names}
            for p in paramnames:
                if p == 'minuslogprior':
                    labels.update({p: '$-\\ln\\pi$'})
                elif 'minuslogprior_' in p:
                    sub = p.split('_', maxsplit=1)[-1].lstrip('_')
                    labels.update({p: f'$-\\ln\\pi_\\mathrm{{{sub}}}$'})
            return paramnames, labels
        except ImportError:
            return paramnames, {}


def read_cobaya(root, *args, **kwargs):
    """Read Cobaya yaml files.

    Note that in order to optimally read chains from Cobaya you need to have
    `GetDist <https://getdist.readthedocs.io/en/latest/>`__ installed.

    Parameters
    ----------
    root : str
        root name for reading files in Cobaya format, i.e. the files
        ``<root>.*.txt`` and ``<root>.updated.yaml``.

    Returns
    -------
    :class:`anesthetic.samples.MCMCSamples`

    """
    dirname, basename = os.path.split(root)

    files = os.listdir(os.path.dirname(root))
    regex = re.escape(basename) + r'.([0-9]+)\.txt'
    matches = [re.match(regex, f) for f in files]
    chain_files = [(m.group(1), os.path.join(dirname, m.group(0)))
                   for m in matches if m]
    if not chain_files:
        raise FileNotFoundError(dirname + '/' + regex + " not found.")
    chain_files.sort(key=lambda chain_file: int(chain_file[0]))

    columns, labels = read_paramnames(root)
    columns = kwargs.pop('columns', columns)
    labels = kwargs.pop('labels', labels)
    kwargs['label'] = kwargs.get('label', os.path.basename(root))

    chain_lengths = [_count_samples(file) for _, file in chain_files]
    nsamples = sum(chain_lengths)
    data = np.empty((nsamples, len(columns)))
    weights = np.empty(nsamples, dtype=int)
    minuslogP = np.empty(nsamples)
    chains = np.empty(nsamples, dtype=int)

    start = 0
    for (i, chain_file), chain_length in zip(chain_files, chain_lengths):
        chain_data = np.loadtxt(chain_file)
        stop = start + chain_length
        weights[start:stop] = chain_data[:, 0]
        minuslogP[start:stop] = chain_data[:, 1]
        data[start:stop] = chain_data[:, 2:]
        chains[start:stop] = int(i) if i else np.nan
        start = stop

    samples = MCMCSamples(data=data, columns=columns, weights=weights,
                          labels=labels, *args, **kwargs)
    samples['logP'] = -minuslogP
    samples.set_label('logP', '$\\ln\\mathcal{P}$')
    samples['logL'] = -samples['chi2'] / 2
    samples.set_label('logL', '$\\ln\\mathcal{L}$')
    samples['chain'] = chains
    samples.root = root
    samples.label = kwargs['label']

    if np.all(samples.chain == samples.chain.iloc[0]):
        samples.drop(columns='chain', inplace=True, level=0)
    else:
        samples.set_label('chain', r'$n_\mathrm{chain}$')

    return samples
