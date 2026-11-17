"""Contract: the entry points refuse a wrong-length triple, NaN/Inf or inadmissible value with ValueError.

Validation lives in the binding (src/pnm_bindings_checks.hpp, used from src/pnm_bindings.cpp); the
message names the function, the parameter, the admissible range with its dimension and the
offending value. Each validated argument is covered twice: every bad value raises ValueError naming
the parameter, and a valid call still returns what it returned before.

origin_zyx and spacing_zyx are narrowed to float32 before the kernels, so they are checked as
float32 (1e39 is not finite there, 1e-50 is not > 0). origin_zyx is a coordinate: finite only, any
sign. The SDF values themselves are not scanned. extract_network_flow(_mpi) are outside this
round (owned by paper A7's branch) and are not covered here.

Run: PYTHONPATH=<build> python -m pytest -q tests/python/test_input_validation.py
"""
import math
import re

import numpy as np
import pytest

import peclet.pnm as pnm

NAN, INF = math.nan, math.inf
N = 8
O = [0.0, 0.0, 0.0]
S = [1.0, 1.0, 1.0]
HAS_MPI = hasattr(pnm, "mpi_block")


def _sdf():
    """One sphere of radius 3.5 in an 8^3 box: one pore."""
    z, y, x = np.meshgrid(*(np.arange(N, dtype=np.float32),) * 3, indexing="ij")
    return (3.5 - np.sqrt((x - 4) ** 2 + (y - 4) ** 2 + (z - 4) ** 2)).astype(np.float32)


def _raises(call, param):
    with pytest.raises(ValueError, match=re.escape(param)):
        call()


# Bad triples of origin_zyx (a coordinate) and spacing_zyx (a cell size), with the parameter the
# message must name.
BAD_ORIGIN = [([0.0, 0.0], "origin_zyx"), ([], "origin_zyx"), ([0.0] * 4, "origin_zyx"),
              ([NAN, 0.0, 0.0], "origin_zyx[0]"), ([0.0, INF, 0.0], "origin_zyx[1]"),
              ([0.0, 0.0, -INF], "origin_zyx[2]"), ([1e39, 0.0, 0.0], "origin_zyx[0]")]
BAD_SPACING = [([1.0, 1.0], "spacing_zyx"), ([], "spacing_zyx"), ([1.0] * 4, "spacing_zyx"),
               ([0.0, 1.0, 1.0], "spacing_zyx[0]"), ([1.0, -1.0, 1.0], "spacing_zyx[1]"),
               ([1.0, 1.0, NAN], "spacing_zyx[2]"), ([INF, 1.0, 1.0], "spacing_zyx[0]"),
               ([1e-50, 1.0, 1.0], "spacing_zyx[0]"), ([1e39, 1.0, 1.0], "spacing_zyx[0]")]


# --- Pore ------------------------------------------------------------------------------------

@pytest.mark.parametrize("args, param", [
    ((NAN, 0, 0, 1), "x"), ((0, INF, 0, 1), "y"), ((0, 0, -INF, 1), "z"),
    ((0, 0, 0, -1), "radius"), ((0, 0, 0, NAN), "radius"), ((0, 0, 0, 1e39), "radius")])
def test_pore_refuses(args, param):
    _raises(lambda: pnm.Pore(*args), f"Pore: {param} must be")


@pytest.mark.parametrize("args", [(1.5, -2.0, 3.0, 0.75), (0, 0, 0, 0)])
def test_pore_accepts(args):
    p = pnm.Pore(*args)
    assert (p.x, p.y, p.z, p.radius) == args


# --- the physical grid of extract_pores / extract_pore_network(_mpi) -------------------------

GRID_FUNCS = {
    "extract_pores": lambda sdf, o, s: pnm.extract_pores(sdf, o, s),
    "extract_pore_network": lambda sdf, o, s: pnm.extract_pore_network(sdf, o, s),
}
if HAS_MPI:
    GRID_FUNCS["extract_pore_network_mpi"] = (
        lambda sdf, o, s: pnm.extract_pore_network_mpi(sdf, [N, N, N], o, s))


@pytest.mark.parametrize("name", sorted(GRID_FUNCS))
@pytest.mark.parametrize("origin, param", BAD_ORIGIN)
def test_grid_refuses_origin(name, origin, param):
    _raises(lambda: GRID_FUNCS[name](_sdf(), origin, S), f"{name}: {param} must be")


@pytest.mark.parametrize("name", sorted(GRID_FUNCS))
@pytest.mark.parametrize("spacing, param", BAD_SPACING)
def test_grid_refuses_spacing(name, spacing, param):
    _raises(lambda: GRID_FUNCS[name](_sdf(), O, spacing), f"{name}: {param} must be")


def test_grid_accepts():
    """A valid physical grid (negative origin, anisotropic spacing) still extracts the one pore."""
    o, s = [-1.0, 2.0, -3.0], [0.5, 0.25, 2.0]
    pores = pnm.extract_pores(_sdf(), o, s)
    fused, seg, conns = pnm.extract_pore_network(_sdf(), o, s)
    assert len(pores) == len(fused) == 1 and seg.shape == (N, N, N)
    assert (pores[0].z, pores[0].y, pores[0].x) == pytest.approx((-1.0 + 4 * 0.5, 2.0 + 4 * 0.25,
                                                                  -3.0 + 4 * 2.0))
    if HAS_MPI:
        mpores, mseg, mconns = pnm.extract_pore_network_mpi(_sdf(), [N, N, N], o, s)
        assert len(mpores) == 1 and np.array_equal(mseg, seg) and np.array_equal(mconns, conns)


# --- segment_volume --------------------------------------------------------------------------

@pytest.mark.parametrize("spacing, param", BAD_SPACING)
def test_segment_volume_refuses(spacing, param):
    _raises(lambda: pnm.segment_volume(_sdf(), spacing), f"segment_volume: {param} must be")


def test_segment_volume_accepts():
    seg = pnm.segment_volume(_sdf(), [0.5, 0.25, 2.0])
    assert seg.shape == (N, N, N) and seg.max() == 1


# --- extract_topology ------------------------------------------------------------------------

@pytest.mark.parametrize("shape, param", [
    ([-1, -1, N ** 3], "shape_zyx[0]"), ([N, -N, N], "shape_zyx[1]"), ([N, N, -1], "shape_zyx[2]"),
    ([2 ** 30, 2 ** 30, 16], "shape_zyx")])
def test_extract_topology_refuses(shape, param):
    seg = pnm.segment_volume(_sdf(), S)
    flat = seg.ravel() if shape[0] != 2 ** 30 else np.zeros(0, np.int32)
    _raises(lambda: pnm.extract_topology(flat, shape), f"extract_topology: {param} must be")


def test_extract_topology_keeps_its_runtime_errors():
    """The shape checks that predate this round stay RuntimeError (length, mismatch, rank)."""
    seg = pnm.segment_volume(_sdf(), S)
    with pytest.raises(RuntimeError, match="shape_zyx must be"):
        pnm.extract_topology(seg.ravel(), [N, N])
    with pytest.raises(RuntimeError, match="does not match"):
        pnm.extract_topology(seg.ravel(), [N, N, N + 1])
    with pytest.raises(RuntimeError, match="or flat"):
        pnm.extract_topology(np.zeros((N, N), np.int32))


def test_extract_topology_accepts():
    seg = pnm.segment_volume(_sdf(), S)
    conns = pnm.extract_topology(seg)
    assert np.array_equal(pnm.extract_topology(seg.ravel(), [N, N, N]), conns)
    assert pnm.extract_topology(np.zeros(0, np.int32), [0, N, N]).shape == (0, 2)


# --- the MPI entry points (a singleton world outside mpirun) ---------------------------------

mpi = pytest.mark.skipif(not HAS_MPI, reason="module built without PECLET_PNM_MPI")


@mpi
@pytest.mark.parametrize("shape, param", [
    ([0, N, N], "global_shape_zyx[0]"), ([N, -1, N], "global_shape_zyx[1]"),
    ([N, N, -N], "global_shape_zyx[2]")])
def test_mpi_block_refuses(shape, param):
    _raises(lambda: pnm.mpi_block(shape), f"mpi_block: {param} must be")


@mpi
def test_mpi_block_keeps_its_runtime_error():
    with pytest.raises(RuntimeError, match="global_shape_zyx must be"):
        pnm.mpi_block([N, N])


@mpi
def test_mpi_block_accepts():
    offset, shape = pnm.mpi_block([N, N, N])
    assert len(offset) == len(shape) == 3 and all(s >= 1 for s in shape)


@mpi
@pytest.mark.parametrize("shape, param", [
    ([N, N], "global_shape_zyx"), ([], "global_shape_zyx"), ([N] * 4, "global_shape_zyx"),
    ([0, N, N], "global_shape_zyx[0]"), ([N, -1, N], "global_shape_zyx[1]")])
def test_extract_pore_network_mpi_refuses_global_shape(shape, param):
    _raises(lambda: pnm.extract_pore_network_mpi(_sdf(), shape, O, S),
            f"extract_pore_network_mpi: {param} must be")
