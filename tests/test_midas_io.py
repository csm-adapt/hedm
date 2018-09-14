__author__ = "Branden Kappes"
__package__ = "hedm"


import pytest
import numpy as np
from hedm.midas.io import Grains
from hedm.midas.io import LatticeParameters_
from hedm.midas.io import Grain_

@pytest.fixture
def minimal():
    """
    Minimal Grains.csv file taken from "data/Grains-minimal.csv".

    :return: Grains object.
    """
    grains = Grains()
    grains.numGrains_ = 1
    grains.beamCenter_ = -20.407685
    grains.beamThickness_ = 2000.000000
    grains.globalPosition_ = 0.000000
    grains.numPhases_ = 1
    grains.spaceGroup_ = 194
    grains.latticeParameters_ = LatticeParameters_(
        a=2.922600, b=2.922600, c=4.670050,
        alpha=90.000000, beta=90.000000, gamma=120.000000)
    g = Grain_()
    g.ID = 28993
    g.orientation = np.array(
        [[ 0.271464, -0.916874,  0.292659],
         [ 0.268031,  0.364068,  0.891972],
         [-0.924374, -0.163697,  0.344582]])
    g.location = np.array([493.292893, 123.973909, -107.835596])
    g.latticeParameters = LatticeParameters_(
        a=2.927856, b=2.928191, c=4.675998,
        alpha=90.009090, beta=89.992938, gamma=119.996361)
    g.diffPos = 34.567103
    g.diffOme = 0.022833
    g.diffAngle = 0.023007
    g.radius = 56.880771
    g.confidence = 0.714286
    g.strainFab = np.array(
        [[ 1866.987953,  -117.463968,   31.632868],
         [ -117.463968,  1371.959916, -218.893846],
         [   31.632868,  -218.893846, 1783.042796]])
    g.strainKen = np.array(
        [[ 1636.729740,  -66.912142,   35.586919],
         [  -66.912142, 1355.049493,  -81.023606],
         [   35.586919,  -81.023606, 2094.456698]])
    g.rmsErrorStrain = 153.000403
    g.phaseNumber = 1
    grains.grains = [g]
    return grains


def test_grains(minimal):
    test = Grains()
    truth = minimal
    test.parse("data/Grains-minimal.csv")
    assert test.NumGrains == truth.NumGrains, \
        f"Number of grains do not match: {test.NumGrains} != {truth.NumGrains}."
    assert np.isclose(test.BeamCenter, truth.BeamCenter), \
        f"Beam centers do not match: {test.BeamCenter} != {truth.BeamCenter}."
    assert np.isclose(test.BeamThickness, truth.BeamThickness), \
        f"Beam thicknesses do not match: " \
        f"{test.BeamThickness} != {truth.BeamThickness}."
    assert np.isclose(test.GlobalPosition, truth.GlobalPosition), \
        f"Global positions do not match: " \
        f"{test.GlobalPositions} != {truth.GlobalPositions}."
    assert test.NumPhases == test.NumPhases, \
        f"Number of phases does not match: " \
        f"{test.NumPhases} != {truth.NumPhases}."
    assert test.SpaceGroup == truth.SpaceGroup, \
        f"Space groups do not match: {test.SpaceGroup} != {truth.SpaceGroup}."
    assert test.LatticeParameters == truth.LatticeParameters, \
        f"Lattice parameters do not match: " \
        f"{test.LatticeParameters} != {truth.LatticeParameters}"
    for i, (lhs, rhs) in enumerate(zip(test.grains, truth.grains)):
        assert lhs == rhs, f"Grain number {i} does not match."