from typing import List

__author__ = 'Branden Kappes <bkappes@mines.edu>'
__project__ = 'hedm'


import os
import numpy as np
import pandas as pd
from io import StringIO


class LatticeParameters_(object):
    def __init__(self, **kwds):
        """
        Lattice parameters.

        :param kwds: Lattice parameters.

        Example:

        lp = LatticeParameters(a=4.09) # returns a cubic lattice

        lp = LatticeParameters(a=3.3, c=4.5, gamma=120) # hexagonal lattice
        """
        if 'a' in kwds:
            self.a = kwds['a']
        if 'b' in kwds:
            self.b = kwds['b']
        if 'c' in kwds:
            self.c = kwds['c']
        if 'alpha' in kwds:
            self.alpha = kwds['alpha']
        if 'beta' in kwds:
            self.beta = kwds['beta']
        if 'gamma' in kwds:
            self.gamma = kwds['gamma']

    def __str__(self) -> str:
        return f"({self.a}, {self.b}, {self.c}, " \
               f"{self.alpha}, {self.beta}, {self.gamma})"

    def __eq__(self, rhs) -> bool:
        """
        Compares two LatticeParameters objects for equality.

        :param rhs: Right hand side
        :type rhs: LatticeParameters object
        :return: True if the two parameters are equal, False otherwise
        :rtype: bool
        """
        return (np.isclose(self.a, rhs.a) and
                np.isclose(self.b, rhs.b) and
                np.isclose(self.c, rhs.c) and
                np.isclose(self.alpha, rhs.alpha) and
                np.isclose(self.beta, rhs.beta) and
                np.isclose(self.gamma, rhs.gamma))

    @property
    def a(self) -> float:
        return getattr(self, "a_", None)

    @a.setter
    def a(self, value):
        self.a_ = float(value)

    @property
    def b(self) -> float:
        return getattr(self, "b_", self.a)

    @b.setter
    def b(self, value):
        self.b_ = float(value)

    @property
    def c(self) -> float:
        return getattr(self, "c_", self.a)

    @c.setter
    def c(self, value):
        self.c_ = float(value)

    @property
    def alpha(self) -> float:
        return getattr(self, "alpha_", None)

    @alpha.setter
    def alpha(self, value):
        self.alpha_ = float(value)

    @property
    def beta(self) -> float:
        return getattr(self, "beta_", self.alpha)

    @beta.setter
    def beta(self, value):
        self.beta_ = float(value)

    @property
    def gamma(self) -> float:
        return getattr(self, "gamma_", self.alpha)

    @gamma.setter
    def gamma(self, value):
        self.gamma_ = float(value)


class Grain_(object):
    def __init__(self):
        """
        Information about the grain.
        """
        self.ID = None
        self.orientation = None
        self.location = None
        self.latticeParameters = None
        self.diffPos = None
        self.diffOme = None
        self.diffAngle = None
        self.radius = None
        self.confidence = None
        self.strainFab = None
        self.strainKen = None
        self.rmsErrorStrain = None
        self.phaseNumber = None

    def __eq__(self, rhs) -> bool:
        """
        Checks for equality; defined by all properties being equal.

        :param rhs: Grain against which to compare.
        :type rhs: Grain_
        :return: True if left (self) and right (rhs) are equal.
        :rtype: bool
        """
        return ((self.ID == rhs.ID) and
                np.allclose(self.orientation, rhs.orientation) and
                np.allclose(self.location, rhs.location) and
                (self.latticeParameters == rhs.latticeParameters) and
                np.isclose(self.diffPos, rhs.diffPos) and
                np.isclose(self.diffOme, rhs.diffOme) and
                np.isclose(self.diffAngle, rhs.diffAngle) and
                np.isclose(self.radius, rhs.radius) and
                np.allclose(self.strainFab, rhs.strainFab) and
                np.allclose(self.strainKen, rhs.strainKen) and
                (self.phaseNumber == rhs.phaseNumber))

    def from_Series(self, series: pd.Series):
        """
        Read grain information from pandas Series data in-place.

        :param series: Grain information read from Grains.csv-formatted file.
        :type series: pandas.Series
        :return: Reference to self
        """
        self.ID = series['GrainID']
        self.orientation = series[['O11', 'O12', 'O13',
                                   'O21', 'O22', 'O23',
                                   'O31', 'O32', 'O33']].values.reshape((3, 3))
        self.location = series[['X', 'Y', 'Z']].values
        self.latticeParameters = LatticeParameters_(
            a=series['a'], b=series['b'], c=series['c'],
            alpha=series['alpha'], beta=series['beta'], gamma=series['gamma'])
        self.diffPos = series['DiffPos']
        self.diffOme = series['DiffOme']
        self.diffAngle = series['DiffAngle']
        self.radius = series['GrainRadius']
        self.confidence = series['Confidence']
        self.strainFab = series[
            ['eFab11', 'eFab12', 'eFab13',
             'eFab21', 'eFab22', 'eFab23',
             'eFab31', 'eFab32', 'eFab33']].values.reshape((3, 3))
        self.strainKen = series[
            ['eKen11', 'eKen12', 'eKen13',
             'eKen21', 'eKen22', 'eKen23',
             'eKen31', 'eKen32', 'eKen33']].values.reshape((3, 3))
        self.rmsErrorStrain = series['RMSErrorStrain']
        self.phaseNumber = int(series['PhaseNr'])
        return self


class Grains(object):
    """
    Stores information about the grains output from MIDAS HEDM.
    """
    grains: List[None]

    def __init__(self):
        self.numGrains_ = None
        self.beamCenter_ = None
        self.beamThickness_ = None
        self.globalPosition_ = None
        self.numPhases_ = None
        self.spaceGroup_ = None
        self.latticeParameters_ = None
        self.grains = []

    @property
    def NumGrains(self) -> int:
        return self.numGrains_

    @property
    def BeamCenter(self) -> float:
        return self.beamCenter_

    @property
    def BeamThickness(self) -> float:
        return self.beamThickness_

    @property
    def GlobalPosition(self) -> float:
        return self.globalPosition_

    @property
    def NumPhases(self) -> int:
        return self.numPhases_

    @property
    def SpaceGroup(self) -> int:
        return self.spaceGroup_

    @property
    def LatticeParameters(self):
        return self.latticeParameters_

    def parse(self, filename: str):
        """
        Reads Grains.csv-formatted file output from MIDAS.

        :param filename: File from which the grains data is read.
        :type filename: string
        :return: Reference to self.
        """
        # check that the file exists
        if not os.path.isfile(filename):
            raise IOError(f'{filename} was not found.')
        # read data from file into a string object
        sio = StringIO()
        with open(filename) as ifs:
            sio.write(ifs.read().replace('%', ''))
            sio.seek(0)
            table = pd.read_table(sio, skiprows=8, index_col=False)
            sio.seek(0)
            header = [s.strip() for s in sio.readlines()[:8]]
        # number of grains
        key, value = header[0].split()
        if key != 'NumGrains':
            IOError(f'{filename} is not a valid format.')
        self.numGrains_ = int(value)
        # beam center
        key, value = header[1].split()
        if key != 'BeamCenter':
            IOError(f'{filename} is not a valid format.')
        self.beamCenter_ = float(value)
        # beam thickness
        key, value = header[2].split()
        if key != 'BeamThickness':
            IOError(f'{filename} is not a valid format.')
        self.beamThickness_ = float(value)
        # global position
        key, value = header[3].split()
        if key != 'GlobalPosition':
            IOError(f'{filename} is not a valid format.')
        self.globalPosition_ = float(value)
        # number of phases
        key, value = header[4].split()
        if key != 'NumPhases':
            IOError(f'{filename} is not a valid format.')
        self.numPhases_ = int(value)
        # phase info: this defines a block of information and is not, at the
        # moment, useful on its own.
        # key = header[5]
        # if key != 'PhaseInfo':
        #     IOError(f'{filename} is not a valid format.')
        # space group
        key, value = header[6].split(':')
        if key != 'SpaceGroup':
            IOError(f'{filename} is not a valid format.')
        self.spaceGroup_ = int(value)
        # lattice parameters
        key, value = header[7].split(':')
        if key != 'Lattice Parameter':
            IOError(f'{filename} is not a valid format.')
        a, b, c, alpha, beta, gamma = value.strip().split(' ')
        self.latticeParameters_ = LatticeParameters_(
            a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        # read data from the table
        self.grains = [Grain_().from_Series(row) for i, row in table.iterrows()]
        # verify the file integrity
        if len(self.grains) != self.NumGrains:
            msg = "{} grains were expected, but only {} were read.".format(
                self.NumGrains, len(self.grains))
            raise IOError(msg)
        return self
