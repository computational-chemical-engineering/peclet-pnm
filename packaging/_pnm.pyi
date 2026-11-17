"""
pnm — pore-network extraction from SDF geometry (Kokkos).

Arrays are (Nz,Ny,Nx) C-order and every describing triple is z-y-x with a `_zyx` suffix. Precision: the SDF is float32 and the geometry kernels compute in float32; origin_zyx / spacing_zyx are doubles narrowed to float32 (pore centres and radii are float32 in the input unit system); the network-flow MAC fields (u, v, w, p, ox, oy, oz) are float64 and their sums are double. SDF sign: negative inside the solid, positive in the pore space.
"""

from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def finalize() -> None:
    """
    Release every live object and zero-copy array of this module, then Kokkos::finalize() (deterministic teardown; also run automatically at interpreter exit). Idempotent. After it, the module's solver objects and *_view arrays must not be used.
    """

execution_space: str

class SDFReader:
    """
    VTI (VTK ImageData) reader for a signed-distance-field volume. Stateless: the class exists only to hold read_vti as a static method, not to be instantiated — call SDFReader.read_vti(filename) directly.
    """

    @staticmethod
    def read_vti(arg: str, /) -> tuple:
        """
        Reads a VTI (VTK ImageData) SDF volume. Returns the tuple (sdf, origin_zyx, spacing_zyx), in that order: sdf a float32 (Nz,Ny,Nx) C-order array over the reader's buffer (no copy), origin_zyx and spacing_zyx the z-y-x lists of the grid origin and cell size (float64).
        """

class Pore:
    """
    One detected pore: a strict local maximum of the SDF over its 26-neighbourhood (periodic in all three directions). The centre is the peak voxel's position plus a squared-SDF-weighted sub-voxel offset over the 3x3x3 stencil, in the unit system of origin_zyx / spacing_zyx; the radius is the SDF value at the peak, i.e. the physical inscribed-sphere radius. All fields are float32. The centre is exposed as three self-named scalars x, y, z (not a triple), so no axis order is implied and no _zyx suffix applies — the documented exception to NAMING.md 1.7.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, x: float, y: float, z: float, radius: float) -> None: ...

    @property
    def x(self) -> float:
        """Centre x coordinate (physical units of origin_zyx / spacing_zyx)."""

    @x.setter
    def x(self, arg: float, /) -> None: ...

    @property
    def y(self) -> float:
        """Centre y coordinate (physical units of origin_zyx / spacing_zyx)."""

    @y.setter
    def y(self, arg: float, /) -> None: ...

    @property
    def z(self) -> float:
        """Centre z coordinate (physical units of origin_zyx / spacing_zyx)."""

    @z.setter
    def z(self, arg: float, /) -> None: ...

    @property
    def radius(self) -> float:
        """
        Inscribed-sphere radius = the SDF value at the peak voxel (physical units).
        """

    @radius.setter
    def radius(self, arg: float, /) -> None: ...

    def __repr__(self) -> str: ...

def extract_pores(sdf: Annotated[NDArray[numpy.float32], dict(order='C')], origin_zyx: Sequence[float], spacing_zyx: Sequence[float]) -> list[Pore]:
    """
    Pore detection: every voxel with sdf > 0 that is a strict local maximum of the SDF over its 26 neighbours (periodic wrap in x, y, z; ties broken towards the higher flat index) becomes a Pore. sdf is a float32 (Nz,Ny,Nx) C-order array; origin_zyx / spacing_zyx are the grid's z-y-x origin and cell size (narrowed to float32). Returns the list of Pore(x, y, z, radius) in the input unit system, in device-completion order (NOT sorted; sort by (z, y, x) for a reproducible order). Capped at 1e6 pores.
    """

def segment_volume(sdf: Annotated[NDArray[numpy.float32], dict(order='C')], spacing_zyx: Sequence[float]) -> NDArray[numpy.int32]:
    """
    Marker-controlled watershed segmentation of the SDF grid. Returns an int32 (Nz,Ny,Nx) C-order label array (sdf.shape; the same memory as the kernels' flat x-fastest label vector — no copy, no reshape needed): pore voxels (sdf > 0) carry the id of the pore basin they belong to, 1, 2, ... in first-encounter (flat-index) order of the basin peaks — the same peaks extract_pores finds — assigned by a gradient-ascent path to the local SDF maximum; solid voxels (sdf <= 0) carry the NEGATIVE id -1, -2, ... of their connected solid grain (26-connected components of the deep solid, sdf < -1.5 * min spacing, flooded outwards by a Jacobi min-label sweep), and 0 marks solid debris no grain reached. Periodic in all three directions. spacing_zyx (the cell size per axis, in the SDF's length units) only sets the deep-solid marker threshold.
    """

def extract_topology(segmentation: Annotated[NDArray[numpy.int32], dict(order='C', writable=False)], shape_zyx: Sequence[int] | None = None) -> NDArray[numpy.int32]:
    """
    Label adjacency of a segment_volume result: the sorted, unique (a, b) pairs with a < b of labels that share a voxel face (+x, +y, +z, periodic wrap), as an (M,2) int32 array. segmentation is the int32 (Nz,Ny,Nx) array segment_volume returned (read in place, no copy); a flat x-fastest label vector is accepted too, then shape_zyx = (Nz, Ny, Nx) of the grid it was made on (= sdf.shape) is required. Rows with a label <= 0 are pore-solid (or grain-grain / debris) contacts; the pore-to-pore throats are the rows with both labels > 0. One row per label pair (a per-PATCH throat list, which can repeat a pair, is what extract_network_flow returns).
    """

def extract_pore_network(sdf: Annotated[NDArray[numpy.float32], dict(order='C')], origin_zyx: Sequence[float], spacing_zyx: Sequence[float]) -> tuple:
    """
    Fused extraction (SDF uploaded once, segmentation device-resident across stages): returns the tuple (pores, segmentation, connections), in that order, each element exactly what extract_pores, segment_volume and extract_topology return (list[Pore]; int32 (Nz,Ny,Nx) array; (M,2) int32 array).

    UNITS: everything is in the system `origin_zyx` / `spacing_zyx` and the SDF are stated in (the VTI's own, straight from SDFReader.read_vti) — pore centres are physical coordinates and a pore radius is the physical inscribed radius, not a voxel count. Pass spacing_zyx = [1,1,1] and origin_zyx = [0,0,0] to work in voxels. Measured on flow/data/packing_ring.vti: doubling the spacing, the origin and the SDF together leaves the pore count (7199) and the throat-connection count (53020) unchanged and doubles every radius BITWISE. One sub-voxel caveat: the centre's intra-cell offset is guarded by an absolute `sw > 1e-6` on a squared-distance weight sum, so a near-degenerate peak's centre can shift by a fraction of a cell (measured up to 0.37 cells) under a change of unit system. Radii, counts and topology are unaffected.
    """

def extract_network_flow(sdf: Annotated[NDArray[numpy.float32], dict(order='C')], origin_zyx: Sequence[float], spacing_zyx: Sequence[float], u: Annotated[NDArray[numpy.float64], dict(order='C')], v: Annotated[NDArray[numpy.float64], dict(order='C')], w: Annotated[NDArray[numpy.float64], dict(order='C')], p: Annotated[NDArray[numpy.float64], dict(order='C')], ox: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, oy: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, oz: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, grad_p_zyx: Sequence[float] = [0.0, 0.0, 0.0]) -> dict:
    """
    Pore-network FLOW data from a MAC field on the same grid as the SDF: segments the SDF, then returns per-pore-label centers/pressures (trilinear p at the basin peak, periodic) and per-throat flow rates (sum of openness-weighted MAC face fluxes o*u*A over the label-interface faces; positive from the lower to the higher label). All arrays are (Nz,Ny,Nx) C-order on the SDF grid: pass flow's fields as get_uf().T, get_p().T, get_ox().T, ... u(i,j,k) is the -x face velocity of cell (i,j,k) (flow's MAC layout); omit ox/oy/oz for a fully open grid. grad_p_zyx adds the macroscopic gradient along the min-image pore-to-pore vector to throat_dp (= P_i - P_j, drives flow i->j when positive). pore_residual is the signed flux sum over each pore's whole boundary — ~solver tolerance when u is flow's projected divergence-free field. Throats are PER-PATCH (a connected patch of interface faces): two disjoint interfaces between the same two pores are separate parallel throats, so the throat list can repeat a label pair.

    Returns a dict: 'pores' (list[Pore], ordered by label: pores[k] is label k+1), 'pore_pressure' and 'pore_residual' (float64 (N,)), 'throats' ((M,2) int32 label pairs a < b, per PATCH), 'throat_flow', 'throat_area' and 'throat_dp' (float64 (M,)).

    Momentum/volume extension (per unit density; a throat is oriented throats[t,0] -> throats[t,1] as throat_flow; every *_zyx array is (count,3) float64 with columns z, y, x): 'throat_momentum_flux_zyx' (M,3) = sum over the patch faces of o*A*u_vec*(u.n), u_vec the face-centre velocity (normal = the MAC face value, each tangential component the mean of the 4 nearest staggered values); 'throat_kinetic_energy_flux' (M,) = sum o*A*|u_vec|^2/2*(u.n); 'throat_pressure_work_flux' (M,) = sum o*A*p_f*(u.n), p_f the mean of the two adjacent cell pressures (periodic part); 'throat_normal_zyx' (M,3) = sum o*A*n / sum o*A (oriented, NOT normalised: |n| < 1 on a curved patch); 'throat_centroid_zyx' (M,3), open-area weighted, in the periodic image nearest pore throats[t,0]'s peak voxel (not wrapped into the box); 'throat_dp_mean' (M,) = throat_dp evaluated with pore_mean_pressure at the volume centroids. Per pore, over the fluid cells (sdf > 0) of its watershed segment: 'pore_volume' (N,), 'pore_mean_pressure' (N,) (periodic part of p, as pore_pressure), 'pore_mean_kinetic_energy' (N,) = mean |u_c|^2/2 and 'pore_mean_velocity_zyx' (N,3) with u_c the cell-centred velocity (mean of the cell's two faces per component), 'pore_volume_centroid_zyx' (N,3) in the image of the pore's peak voxel.

    PRECONDITION (cross-repo, not checkable here): the openness arrays must be the face openness the velocity field was projected with. For peclet.flow's cut-cell IBM that means the solid was set with set_solid(..., cutcell_pressure=True) — without it every openness flow reports is 0, so every throat_flow and throat_area comes back 0; for the ghost-cell IBM (set_ghost_projection) pass get_ox_proj()/get_oy_proj()/get_oz_proj() instead.
    """

def mpi_block(global_shape_zyx: Sequence[int]) -> tuple:
    """
    This rank's ORB block of the global (Nz,Ny,Nx) grid (collective; MPI_Init is called if needed): returns the tuple (offset_zyx, shape_zyx), in that order — two int 3-tuples, the block's first VOXEL index per axis and its voxel count per axis. offset_zyx is an integer grid offset, NOT the physical origin_zyx the extraction functions take (that stays the global grid's). Slice the global SDF as sdf[oz:oz+sz, oy:oy+sy, ox:ox+sx] and pass the block to extract_pore_network_mpi / extract_network_flow_mpi.
    """

def extract_pore_network_mpi(sdf_local: Annotated[NDArray[numpy.float32], dict(order='C')], global_shape_zyx: Sequence[int], origin_zyx: Sequence[float], spacing_zyx: Sequence[float]) -> tuple:
    """
    Distributed fused extraction (collective over MPI_COMM_WORLD). sdf_local is this rank's ORB block (Nz,Ny,Nx C-order, from mpi_block); origin_zyx / spacing_zyx are the GLOBAL grid's, in the SDF's length units (as for extract_pore_network). Returns the tuple (pores, segmentation, connections), in that order: the pores whose peak this rank owns (list[Pore]), this rank's block of the segmentation (int32 array of sdf_local.shape, global label ids), and the GLOBAL connection list ((M,2) int32, identical on every rank). Bit-exact to the single-rank extract_pore_network on the gathered grid.
    """

def extract_network_flow_mpi(sdf_local: Annotated[NDArray[numpy.float32], dict(order='C')], global_shape_zyx: Sequence[int], origin_zyx: Sequence[float], spacing_zyx: Sequence[float], u: Annotated[NDArray[numpy.float64], dict(order='C')], v: Annotated[NDArray[numpy.float64], dict(order='C')], w: Annotated[NDArray[numpy.float64], dict(order='C')], p: Annotated[NDArray[numpy.float64], dict(order='C')], ox: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, oy: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, oz: Annotated[NDArray[numpy.float64], dict(order='C')] | None = None, grad_p_zyx: Sequence[float] = [0.0, 0.0, 0.0]) -> dict:
    """
    Distributed network-flow extraction (collective over MPI_COMM_WORLD): all arrays are this rank's ORB block (Nz,Ny,Nx C-order, from mpi_block; MAC fields from a distributed peclet.flow run on the SAME BlockDecomposer). Returns the same dict as extract_network_flow, GLOBAL and identical on every rank.

    Returns a dict: 'pores' (list[Pore], ordered by label: pores[k] is label k+1), 'pore_pressure' and 'pore_residual' (float64 (N,)), 'throats' ((M,2) int32 label pairs a < b, per PATCH), 'throat_flow', 'throat_area' and 'throat_dp' (float64 (M,)).

    Momentum/volume extension (per unit density; a throat is oriented throats[t,0] -> throats[t,1] as throat_flow; every *_zyx array is (count,3) float64 with columns z, y, x): 'throat_momentum_flux_zyx' (M,3) = sum over the patch faces of o*A*u_vec*(u.n), u_vec the face-centre velocity (normal = the MAC face value, each tangential component the mean of the 4 nearest staggered values); 'throat_kinetic_energy_flux' (M,) = sum o*A*|u_vec|^2/2*(u.n); 'throat_pressure_work_flux' (M,) = sum o*A*p_f*(u.n), p_f the mean of the two adjacent cell pressures (periodic part); 'throat_normal_zyx' (M,3) = sum o*A*n / sum o*A (oriented, NOT normalised: |n| < 1 on a curved patch); 'throat_centroid_zyx' (M,3), open-area weighted, in the periodic image nearest pore throats[t,0]'s peak voxel (not wrapped into the box); 'throat_dp_mean' (M,) = throat_dp evaluated with pore_mean_pressure at the volume centroids. Per pore, over the fluid cells (sdf > 0) of its watershed segment: 'pore_volume' (N,), 'pore_mean_pressure' (N,) (periodic part of p, as pore_pressure), 'pore_mean_kinetic_energy' (N,) = mean |u_c|^2/2 and 'pore_mean_velocity_zyx' (N,3) with u_c the cell-centred velocity (mean of the cell's two faces per component), 'pore_volume_centroid_zyx' (N,3) in the image of the pore's peak voxel.

    PRECONDITION (cross-repo, not checkable here): the openness arrays must be the face openness the velocity field was projected with. For peclet.flow's cut-cell IBM that means the solid was set with set_solid(..., cutcell_pressure=True) — without it every openness flow reports is 0, so every throat_flow and throat_area comes back 0; for the ghost-cell IBM (set_ghost_projection) pass get_ox_proj()/get_oy_proj()/get_oz_proj() instead.
    """
