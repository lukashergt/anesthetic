"""Tools for reading from getdist chains files."""
import sys
import os
import warnings
import numpy
import glob
from anesthetic.read.chainreader import ChainReader
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import getdist
        from getdist import loadMCSamples
except ImportError:
    print('getdist not imported')

class GetDistReader(ChainReader):
    """Read getdist files."""

    def paramnames(self):
        r"""Read <root>.paramnames in getdist format.

        This file should contain one or two columns. The first column indicates
        a reference name for the sample, used as labels in the pandas array.
        The second optional column should include the equivalent axis label,
        possibly in tex, with the understanding that it will be surrounded by
        dollar signs, for example

        <root.paramnames>

        a1     a_1
        a2     a_2
        omega  \omega
        """
        if os.path.isfile(self.yaml_file):
            with open(self.root + ".1.txt") as f:
                header = f.readline()[1:]
            paramnames = header.split()[2:]
            s = loadMCSamples(file_root=self.root)
            tex = {i.name: '$' + i.label + '$' for i in s.paramNames.names}
            return paramnames, tex
        elif os.path.split(os.path.split(self.root)[0])[-1] == 'raw_polychord_output':
            raw_dir, root_name = os.path.split(self.root)
            chain_dir, _ = os.path.split(raw_dir)

            with open(chain_dir + '/' + root_name + ".1.txt") as f:
                header = f.readline()[1:]
            paramnames_long = header.split()
            paramnames_long.remove('minuslogprior')
            paramnames_long.remove('chi2')
            paramnames = [p + '/-2' if 'chi2' in p and 'CMB' not in p else p
                          for p in paramnames_long][2:]

            s = loadMCSamples(file_root=chain_dir + '/' + root_name)
            # tex = {i.name: '$' + i.label + '$' for i in s.paramNames.names
            #        if i.name in paramnames}
            # paramnames = [i.name for i in s.paramNames.names]
            tex = {i.name: '$' + i.label + '$' for i in s.paramNames.names}
            # paramnames = paramnames[:-2]
            return paramnames, tex
        else:
            try:
                with open(self.paramnames_file, 'r') as f:
                    paramnames = []
                    tex = {}
                    for line in f:
                        line = line.strip().split()
                        paramname = line[0].replace('*', '')
                        paramnames.append(paramname)
                        if len(line) > 1:
                            tex[paramname] = '$' + ' '.join(line[1:]) + '$'
                    return paramnames, tex
            except IOError:
                return super(GetDistReader, self).paramnames()

    def limits(self):
        """Read <root>.ranges in getdist format."""
        if os.path.isfile(self.yaml_file):
            s = loadMCSamples(file_root=self.root)
            limits = {i: (s.ranges.getLower(i), s.ranges.getUpper(i))
                      for i in s.ranges.names}
            return limits
        elif os.path.split(os.path.split(self.root)[0])[-1] == 'raw_polychord_output':
            raw_dir, root_name = os.path.split(self.root)
            chain_dir, _ = os.path.split(raw_dir)
            s = loadMCSamples(file_root=chain_dir + '/' + root_name)
            limits = {i: (s.ranges.getLower(i), s.ranges.getUpper(i))
                      for i in s.ranges.names}
            return limits
        else:
            try:
                with open(self.ranges_file, 'r') as f:
                    limits = {}
                    for line in f:
                        line = line.strip().split()
                        paramname = line[0]
                        try:
                            xmin = float(line[1])
                        except ValueError:
                            xmin = None
                        try:
                            xmax = float(line[2])
                        except ValueError:
                            xmax = None
                        limits[paramname] = (xmin, xmax)
                    return limits
            except IOError:
                return super(GetDistReader, self).limits()

    def samples(self):
        """Read <root>_1.txt in getdist format."""
        data = numpy.concatenate([numpy.loadtxt(chains_file)
                                  for chains_file in self.chains_files])
        weights, chi2, samples = numpy.split(data, [1, 2], axis=1)
        logL = chi2/-2.
        return weights.flatten(), logL.flatten(), samples

    @property
    def paramnames_file(self):
        """File containing parameter names."""
        return self.root + '.paramnames'

    @property
    def yaml_file(self):
        """Cobaya parameter file."""
        return self.root + '.updated.yaml'

    @property
    def ranges_file(self):
        """File containing parameter names."""
        return self.root + '.ranges'

    @property
    def chains_files(self):
        """File containing parameter names."""
        files = glob.glob(self.root + '_[0-9].txt')
        if not files:
            files = glob.glob(self.root + '.[0-9].txt')
        if not files:
            files = [self.root + '.txt']

        return files
