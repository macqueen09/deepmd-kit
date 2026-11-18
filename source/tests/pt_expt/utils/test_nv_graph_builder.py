# SPDX-License-Identifier: LGPL-3.0-or-later
"""nvalchemiops carry-all NeighborGraph builder: neighbor SET must equal the
in-tree ``dense`` carry-all reference. CUDA + nvalchemi-toolkit-ops only.
"""

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)

from ...common.dpmodel.test_neighbor_graph import (
    assert_csr_consistent,
)

nv_builder = pytest.importorskip("deepmd.pt_expt.utils.nv_graph_builder")
from deepmd.pt.utils.nv_nlist import (
    is_nv_available,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_nv_available()),
    reason="nvalchemiops requires CUDA + nvalchemi-toolkit-ops",
)


def _edge_multiset(ng):
    edge_index = np.asarray(ng.edge_index.cpu())
    edge_vec = np.asarray(ng.edge_vec.detach().cpu())
    edge_mask = np.asarray(ng.edge_mask.cpu())
    return sorted(
        (
            int(edge_index[0, edge]),
            int(edge_index[1, edge]),
            tuple(np.round(edge_vec[edge], 10)),
        )
        for edge in range(edge_index.shape[1])
        if edge_mask[edge]
    )


@pytest.mark.parametrize("periodic", [False, True])  # non-PBC and PBC
def test_nv_matches_intree_carry_all(periodic):
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
        device=dev,
    )
    box = (
        (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
        if periodic
        else None
    )
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64, device=dev)
    ng_ref = build_neighbor_graph(coord, atype, box, 2.0)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    assert _edge_multiset(ng) == _edge_multiset(ng_ref)


def test_nv_batches_frames_without_python_loop():
    """NV batches flattened training coordinates without a per-frame loop."""
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    coord = torch.tensor(
        rng.random((3, 5, 3)) * 3.0, dtype=torch.float64, device=dev
    ).reshape(3, -1)
    box = (
        (torch.eye(3, dtype=torch.float64, device=dev) * 4.0)
        .reshape(1, 3, 3)
        .repeat(3, 1, 1)
    )
    atype = torch.tensor(
        [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0]],
        dtype=torch.int64,
        device=dev,
    )
    ng_ref = build_neighbor_graph(coord, atype, box, 2.0)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    assert _edge_multiset(ng) == _edge_multiset(ng_ref)


@pytest.mark.parametrize("edges", [0, 1, 33, 8192, 8193])
@pytest.mark.parametrize("store_fp32", [False, True])
def test_fused_periodic_geometry_matches_eager_bits(
    edges: int, store_fp32: bool
) -> None:
    """Use the original broadcast sum as the independent FP64 reference."""
    generator = torch.Generator(device="cuda").manual_seed(17)
    coord = torch.randn(11, 3, dtype=torch.float64, device="cuda", generator=generator)
    cell = torch.randn(2, 3, 3, dtype=torch.float64, device="cuda", generator=generator)
    endpoints = torch.arange(edges, device="cuda")
    center, source, frame = endpoints % 11, (endpoints + 1) % 11, endpoints % 2
    shift = torch.randint(
        -2, 3, (edges, 3), device="cuda", generator=generator
    ).double()
    expected = (coord[source] - coord[center]) + (shift[:, :, None] * cell[frame]).sum(
        1
    )
    actual = nv_builder._compiled_periodic_edge_vectors(
        coord, center, source, shift, cell, frame, store_fp32=store_fp32
    )
    if store_fp32:
        expected = expected.float()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_nv_edge_vec_is_differentiable():
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
        device=dev,
    ).requires_grad_(True)
    box = (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64, device=dev)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    (ng.edge_vec**2).sum().backward()
    assert coord.grad is not None and torch.any(coord.grad != 0)


@pytest.mark.parametrize("edges", [0, 1, 17])
def test_direct_canonical_nv_matches_generic_converter(edges: int) -> None:
    """Keep all seven canonical fields and the small-edge guard ABI."""
    from dataclasses import (
        fields,
    )

    from deepmd.pt_expt.utils.canonical_graph import (
        canonical_graph_from_pairs,
    )

    n_node = torch.tensor([4, 4], device="cuda")
    center = torch.arange(edges, device="cuda") // 3
    source = torch.arange(edges, device="cuda").flip(0) % 8
    vectors = torch.arange(edges * 3, device="cuda", dtype=torch.float64).reshape(-1, 3)
    reference = canonical_graph_from_pairs(center, source, vectors, n_node, 8)
    actual = nv_builder._canonical_from_ordered_nv(center, source, vectors, n_node, 8)
    for field in fields(reference):
        torch.testing.assert_close(
            getattr(actual, field.name), getattr(reference, field.name), rtol=0, atol=0
        )


def test_nv_excludes_virtual_atoms_like_dense():
    """Virtual atoms (atype < 0) excluded as center AND neighbor (dense contract)."""
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0], [1.8, 1.8, 0.0]]],
        dtype=torch.float64,
        device=dev,
    )
    box = (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
    atype = torch.tensor([[0, -1, 0, 1]], dtype=torch.int64, device=dev)  # 1 virtual
    ng_ref = build_neighbor_graph(coord, atype, box, 2.0)
    ng = nv_builder.build_neighbor_graph_nv(coord, atype, box, 2.0)
    assert _edge_multiset(ng) == _edge_multiset(ng_ref)
    ei = np.asarray(ng.edge_index.cpu())[:, np.asarray(ng.edge_mask.cpu())]
    at = atype.reshape(-1).cpu().numpy()
    assert np.all(at[ei[0]] >= 0) and np.all(at[ei[1]] >= 0)


@pytest.mark.parametrize("csr_mode", ["none", "views", "canonical"])
@pytest.mark.parametrize("exclude_pairs", [False, True])
def test_nv_pair_exclusion_matches_dense(csr_mode: str, exclude_pairs: bool) -> None:
    dev = torch.device("cuda")
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 1.1, 0.0], [1.2, 1.0, 0.0]]],
        dtype=torch.float64,
        device=dev,
    )
    atype = torch.tensor([[0, 1, 0, 1]], dtype=torch.int64, device=dev)
    box = (torch.eye(3, dtype=torch.float64, device=dev) * 3.0).reshape(1, 3, 3)
    pair_excl = PairExcludeMask(2, [(0, 1), (1, 0)]) if exclude_pairs else None
    expected = build_neighbor_graph(
        coord,
        atype,
        box,
        2.0,
        pair_excl=pair_excl,
    )
    actual = nv_builder.build_neighbor_graph_nv(
        coord,
        atype,
        box,
        2.0,
        pair_excl=pair_excl,
        with_csr=csr_mode != "none",
        canonicalize=csr_mode == "canonical",
    )
    assert _edge_multiset(actual) == _edge_multiset(expected)
    if csr_mode != "none":
        assert_csr_consistent(actual)
