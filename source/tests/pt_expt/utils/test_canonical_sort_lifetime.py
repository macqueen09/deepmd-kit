# SPDX-License-Identifier: LGPL-3.0-or-later
"""Destination sort temporaries can expire before the independent source sort."""

import weakref

import pytest
import torch

from deepmd.pt_expt.utils.canonical_graph import (
    canonical_graph_from_pairs,
    canonical_index_values,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 1, 5, 33])
def test_destination_workspace_expires_before_source_sort(monkeypatch, device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    center = torch.arange(edges, device=device) % 3
    source = (center + 1) % 3
    vectors = torch.arange(edges * 3, dtype=torch.float64, device=device).reshape(-1, 3)
    originals = [x.clone() for x in (center, source, vectors)]
    expected_order = torch.argsort(center, stable=True)
    expected_source = source[expected_order]
    expected_vectors = vectors[expected_order].float()
    references = []
    argsort, searchsorted = torch.argsort, torch.searchsorted

    def sort(keys, *args, **kwargs):
        if references:
            assert len(references) == 2
            assert all(ref() is None for ref in references)
        order = argsort(keys, *args, **kwargs)
        if not references:
            references.append(weakref.ref(order))
        return order

    def search(keys, *args, **kwargs):
        if len(references) == 1:
            references.append(weakref.ref(keys))
        return searchsorted(keys, *args, **kwargs)

    monkeypatch.setattr(torch, "argsort", sort)
    monkeypatch.setattr(torch, "searchsorted", search)
    graph = canonical_graph_from_pairs(
        center, source, vectors, torch.tensor([3], device=device), 3
    )
    assert torch.equal(canonical_index_values(graph.source)[:edges], expected_source)
    assert torch.equal(graph.edge_vec[:edges], expected_vectors)
    assert int(graph.destination_row_ptr[-1]) == edges
    assert int(graph.source_row_ptr[-1]) == edges
    for actual, original in zip((center, source, vectors), originals, strict=True):
        assert torch.equal(actual, original)
