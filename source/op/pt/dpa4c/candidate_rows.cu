// SPDX-License-Identifier: LGPL-3.0-or-later
//
// Cached candidate rows for the compact DPA4C canonical graph.
//
// A rebuild of the neighbor cache stores the candidate pairs of rcut + skin
// as destination-major rows: the row pointer, the source of every candidate
// in the order of its row, and the candidate shift pre-multiplied with the
// reference cell. A hot step then reads the rows once: every candidate gets
// its FP64 geometry from the current positions and the strict r^2 < rcut^2
// test of the reference search, a warp counts the kept edges of its row, and
// a second warp compacts the kept edges in the order of the row into storage
// sized by the kept edges. The kernels follow the row count / scan / fill
// shape of the further-block EdgeGraph selection in graph_compress.cu, with
// double geometry and the strict bound in place of its widened float bound.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/types.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

constexpr int kRowThreads = 256;
constexpr int kRowWarps = kRowThreads / 32;

// The kept flags of one hot step, shared by the counting and the compaction
// kernels so the geometry is recomputed only where it is written.
template <typename index_t>
__global__ void __launch_bounds__(kRowThreads)
candidate_count_kernel(const long* __restrict__ row_ptr,
                       const index_t* __restrict__ cand_source,
                       const double* __restrict__ shift_vec,
                       const double* __restrict__ positions,
                       long rows,
                       double bound2,
                       unsigned char* __restrict__ keep,
                       long* __restrict__ counts) {
  const long row = static_cast<long>(blockIdx.x) * kRowWarps + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % 32;
  const double px = positions[row * 3 + 0];
  const double py = positions[row * 3 + 1];
  const double pz = positions[row * 3 + 2];
  int count = 0;
  for (long p = row_ptr[row] + lane; p < row_ptr[row + 1]; p += 32) {
    const long src = static_cast<long>(cand_source[p]) * 3;
    const double vx = positions[src + 0] + shift_vec[p * 3 + 0] - px;
    const double vy = positions[src + 1] + shift_vec[p * 3 + 1] - py;
    const double vz = positions[src + 2] + shift_vec[p * 3 + 2] - pz;
    const bool inside = (vx * vx + vy * vy + vz * vz) < bound2;
    keep[p] = inside ? 1 : 0;
    count += inside ? 1 : 0;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    count += __shfl_xor_sync(0xffffffffu, count, offset);
  }
  if (lane == 0) {
    counts[row + 1] = count;
  }
}

// The kept edges of every row in the order of the row, into storage sized by
// the count of the same step. The geometry is recomputed with the same double
// expression as the counting kernel and narrowed to float once, on write.
template <typename index_t>
__global__ void __launch_bounds__(kRowThreads)
candidate_compact_kernel(const long* __restrict__ row_ptr,
                         const index_t* __restrict__ cand_source,
                         const double* __restrict__ shift_vec,
                         const double* __restrict__ positions,
                         long rows,
                         const unsigned char* __restrict__ keep,
                         const long* __restrict__ new_row_ptr,
                         index_t* __restrict__ out_source,
                         float* __restrict__ out_vec) {
  const long row = static_cast<long>(blockIdx.x) * kRowWarps + threadIdx.x / 32;
  if (row >= rows) {
    return;
  }
  const int lane = threadIdx.x % 32;
  const unsigned before = (1u << lane) - 1u;
  const double px = positions[row * 3 + 0];
  const double py = positions[row * 3 + 1];
  const double pz = positions[row * 3 + 2];
  long target = new_row_ptr[row];
  for (long first = row_ptr[row]; first < row_ptr[row + 1]; first += 32) {
    const long p = first + lane;
    const bool inside = p < row_ptr[row + 1] && keep[p];
    const unsigned kept = __ballot_sync(0xffffffffu, inside);
    if (inside) {
      const long slot = target + __popc(kept & before);
      const long src = static_cast<long>(cand_source[p]) * 3;
      out_source[slot] = cand_source[p];
      out_vec[slot * 3 + 0] =
          static_cast<float>(positions[src + 0] + shift_vec[p * 3 + 0] - px);
      out_vec[slot * 3 + 1] =
          static_cast<float>(positions[src + 1] + shift_vec[p * 3 + 1] - py);
      out_vec[slot * 3 + 2] =
          static_cast<float>(positions[src + 2] + shift_vec[p * 3 + 2] - pz);
    }
    target += __popc(kept);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
candidate_rows_select(torch::Tensor row_ptr,
                                                 torch::Tensor cand_source,
                                                 torch::Tensor shift_vec,
                                                 torch::Tensor positions,
                                                 double rcut) {
  const long rows = row_ptr.numel() - 1;
  const long candidates = cand_source.numel();
  auto counts = torch::zeros({rows + 1}, row_ptr.options());
  if (rows == 0) {
    // 空图: 两个零槽守住"存储至少两槽"的契约, 行指针长度 1。
    return {counts,
            torch::zeros({2}, cand_source.options()),
            torch::zeros({2, 3}, positions.options().dtype(torch::kFloat32))};
  }
  auto keep = torch::empty({candidates},
                           row_ptr.options().dtype(torch::kUInt8));
  const int blocks =
      static_cast<int>((rows + kRowWarps - 1) / kRowWarps);
  auto stream = at::cuda::getCurrentCUDAStream();
  const double bound2 = rcut * rcut;
  if (cand_source.scalar_type() == torch::kInt64) {
    candidate_count_kernel<int64_t>
        <<<blocks, kRowThreads, 0, stream>>>(
            row_ptr.data_ptr<long>(), cand_source.data_ptr<int64_t>(),
            shift_vec.data_ptr<double>(), positions.data_ptr<double>(), rows,
            bound2, keep.data_ptr<unsigned char>(), counts.data_ptr<long>());
  } else if (cand_source.scalar_type() == torch::kUInt32) {
    candidate_count_kernel<uint32_t>
        <<<blocks, kRowThreads, 0, stream>>>(
            row_ptr.data_ptr<long>(),
            static_cast<const uint32_t*>(cand_source.data_ptr()),
            shift_vec.data_ptr<double>(), positions.data_ptr<double>(), rows,
            bound2, keep.data_ptr<unsigned char>(), counts.data_ptr<long>());
  } else {
    TORCH_CHECK(false,
                "dpa4c_candidate_rows_select: cand_source must be int64 or "
                "uint32");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto new_row_ptr = counts.cumsum(0);
  // 有效边数要落主机才能分配精确存储: 一次小同步(热步的百分之几),
  // 换来输出与 canonical_graph_from_pairs 完全相同的存储语义。
  const long valid = new_row_ptr[-1].item<long>();
  const long storage = std::max(valid, 2L);
  auto out_source = torch::empty({storage}, cand_source.options());
  auto out_vec = torch::empty({storage, 3},
                              positions.options().dtype(torch::kFloat32));
  if (storage > valid) {
    // 少于两条边时补零槽, 与 from_pairs 的 guard 语义一致。
    out_source.narrow(0, valid, storage - valid).zero_();
    out_vec.narrow(0, valid, storage - valid).zero_();
  }
  if (cand_source.scalar_type() == torch::kInt64) {
    candidate_compact_kernel<int64_t>
        <<<blocks, kRowThreads, 0, stream>>>(
            row_ptr.data_ptr<long>(), cand_source.data_ptr<int64_t>(),
            shift_vec.data_ptr<double>(), positions.data_ptr<double>(), rows,
            keep.data_ptr<unsigned char>(), new_row_ptr.data_ptr<long>(),
            out_source.data_ptr<int64_t>(), out_vec.data_ptr<float>());
  } else {
    candidate_compact_kernel<uint32_t>
        <<<blocks, kRowThreads, 0, stream>>>(
            row_ptr.data_ptr<long>(),
            static_cast<const uint32_t*>(cand_source.data_ptr()),
            shift_vec.data_ptr<double>(), positions.data_ptr<double>(), rows,
            keep.data_ptr<unsigned char>(), new_row_ptr.data_ptr<long>(),
            static_cast<uint32_t*>(out_source.data_ptr()),
            out_vec.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {new_row_ptr, out_source, out_vec};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(deepmd, library) {
  library.impl("dpa4c_candidate_rows_select", torch::kCUDA,
               &candidate_rows_select);
}
