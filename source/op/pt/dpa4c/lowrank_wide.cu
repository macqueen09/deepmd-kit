// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Explicit instantiations of the descriptor scans whose weights certify a
// low-rank basis wider than the channel-derived default: one hundred
// twenty-eight channels at rank forty-eight, the width the released Plus
// weights certify. The kernels are those of graph_compress_kernel.cuh with
// the basis width carried through the launch chain; this unit only names
// the combinations the dispatch may select, beside the generated shards
// that carry the default width of every channel count.

#include "graph_compress_kernel.cuh"

namespace deepmd_dpa4c {

template void launch_forward<128, 3, 48>(const Arguments& arguments,
                                         cudaStream_t stream);
template void launch_backward<128, 3, 48>(const Arguments& arguments,
                                          cudaStream_t stream);

}  // namespace deepmd_dpa4c
