# ECS「边批流式反向」：导师汇报、代码导读与研究价值评估

日期：2026-10-19。用途：个人 fork 的 Draft 代码 review 与导师汇报，**不是通用生产验收报告**。

本PR是湖南大学提交的。

## 0. 一分钟汇报

我们在正式 DPA4-Neo 的 SO(2) mixing 反向计算中，利用“不同边之间没有数据依赖”的性质，将“每层遍历全部边”改为“每批边连续算完全部反向层”。矩阵公式、严格 FP32 和原 GEMM 输出列并行保持不变，部分临时梯度从全图大小缩小为固定边批大小。

2026-10-18 的实验原型在 RTX 5090 上，N1000 的整模型能量/力/位力 API 时间下降约 3.50%，完整窗口峰值 allocated 显存下降约 8.70%；扣除已有 SO2 inplace 优化后，新增观测为时间下降约 3.41%、峰值下降约 2.06%。小体系 N64 时间下降约 12.6%～14.3%；N4096 速度变化仅约 2.24%，且严格精度尚未全过，不能作为正式成功验收。

这证明了一个有实际价值的调度机会，但**目前不证明“发现了 PyTorch 普遍缺陷”，也不足以单凭这个案例主张一篇 CCF-B 创新论文**。研究方向应是自动识别可分离维度、保持高效原生核、联合规划调度与张量生命周期，而不是仅发表一段手写分批循环。

历史实验与本次提交必须分开：上述性能来自 10 月 18 日字符串适配原型；本 PR 改为显式原生内核步长参数与独立调度模块，移除动态生成/改写源码。本次移植测试及新整模 smoke 见文末，不把旧数据冒充新提交复测。

**本次移植复测的重要负面结果：** 2026-10-19 N64各一进程，原生12.172306ms、移植版13.123442ms，时间增加7.814%；全窗口allocated仍从0.498200降至0.451064GB（-9.461%）。因此今日只复现了该规模的省显存，没有复现历史小体系加速。不能把前一天的18.45ms基线与今天13.12ms候选跨批次算加速，也不能在没有profile/反序重复前认定原因是Torch、CPU或移植寻址。它进一步说明当前必须默认关闭并保留Draft。

## 1. 背景：到底在算什么

模型：`DPA4-Neo-OMat24-v20260627.pt`，SHA256：

```text
8d8b2b02b9575cb9e0af420cc1e8b1d9ffcbc04752c2203d29f48ccd0cd5f07d
```

单个 stack 的正式结构：F=2、Cf=32、lmax=3、ROW=320；m=0 宽128，m=±1宽192；门控宽96；两层 gated residual + 一层 linear residual。两个 interaction block 中各有一个 stack。

这些维度是**此冻结模型的架构常量**，不是所有 SO(2) 模型都必须相同的物理常数。输入原子规模和边数会变化。本实验不通过改变架构、权重或角度截断换取速度。

推理仍需反向：先预测能量，再对位置求导得到力；这不是更新权重的训练。ECS 当前只接管已有 dispatcher 的冻结权重、一阶反向分支。权重梯度、保存状态的导数和 create_graph 继续走原训练/高阶路径。

主要张量：

| 张量 | 典型布局 | 生命周期/作用 |
| --- | --- | --- |
| grad_out | `[E,F,ROW]` | 来自后续模块的上游梯度，只读 |
| z_all | `[2,F,E,ROW]` | 前向保存的门控层预激活，仍全图保存 |
| alpha | `[E,F]` | 每条边每个 focus 的竞争权重，只读 |
| g | `[F,E,ROW]` 或 ECS `[F,C,ROW]` | 当前反向层输出梯度 |
| gz | 同上 | 激活反向后的矩阵输入，不能与 GEMM 输出混用 |
| sig | `[F,E,96]` 或 ECS `[F,C,96]` | 门控 sigmoid 重算暂存 |
| grad_u0 | `[F,E,ROW]` | 必须返回给前一模块的完整梯度 |

ECS 不删除完整 grad_out、z_all、grad_u0，不减少边数，不改变整图邻居关系。

## 2. 如何找到这个优化

### 2.1 从性能现象而非理论峰值出发

前期 GEMM 五参数扫描收益很有限；旧 Nsight 数据提示若干值路径内核存在明显数据移动成本。但旧 profile 的采集范围和精度口径不完全齐全，只能指路，不能用来宣称 ECS 已提升 L2 命中率。

于是从“单个 GEMM 更快”转向“相邻层之间到底保留/写回多少数据”。参考 Megakernels 的有界工作集、沿依赖链执行思想，检查了 stack 的实际依赖。

### 2.2 先试更强融合，失败后调整边界

最初 PBS 试图让一个 CTA 持久执行多层。E约1万时有局部改善，但 E158002 的12组配置全部退化：原生反向约5.047ms，最好PBS约8.780ms。部分配置寄存器 spill 严重；零 spill 组也不快，因为一个 CTA 串行承担了多个输出列 tile。

结论不是“融合无效”，而是不能为了局部存储破坏原来高效的矩阵列并行。ECS 因而保留原内核与输出列 tile，只改变处理不同边批和不同层的先后关系。

### 2.3 识别可交换的两个循环

对单条边 e，可把 gated 反向概括为：

```text
gz[e] = activation_backward(g[e], z[e], gate_weights)
g_prev[e] = g[e] + gz[e] @ W_transposed
```

矩阵乘法会沿特征维收缩，但不会把边 e 与其他边相加。alpha 梯度也在同一边内收缩特征。因此，stack 内部的边批循环与反向层循环可以交换。

```python
# 原调度：中间变量覆盖所有边
g = final_backward(all_edges)
for layer in reversed(gated_layers):
    gz = point_backward(g, z[layer])
    g = gemm_backward(gz, residual=g)

# ECS：每个边批仍按原依赖顺序计算
for edges in cohorts(all_edges, capacity=8192):
    g = final_backward(edges)
    for layer in reversed(gated_layers):
        gz = point_backward(g, z[layer, edges])
        g = gemm_backward(gz, residual=g)
    write_full_result(edges, g)
```

类比：原来每道工序先加工全仓库零件，再进入下一道工序；现在每一小托盘连续走完工序，工位周转箱只需装下一托盘。仓库和最终成品仍存在，减少的是同时周转的中间件。

该交换只在 stack 内成立。图聚合有跨边归约，不能据此直接把完整模型按原子拆开。

## 3. 微观改动：读代码时看哪里

### 3.1 原生文件：保持默认行为，只扩展寻址

文件：`deepmd/pt_expt/kernels/triton/sezm/so2_value_path.py`。

- `_stack_recompute_kernel`、`_stack_point_bwd_kernel`：新增 `saved_edges=0`。默认仍以当前 n_edge 为 z 的 focus/layer 步长；ECS 使用完整 E，但 kernel 的边掩码仍是本批 count。
- `_stack_gemm_bwd_kernel`：新增 `output_edges=0`。最后一层反向输出写回完整 `[F,E,ROW]`，中间层仍写有界工作区。
- `_mixing_stack_backward`：只在训练/高阶分支之后选择候选；默认环境开关关闭，支持范围外沿用原算子。

寻址不能简单全部换成批大小，否则会把第二个 focus、第二个 gated layer 的 z 读错。三个维度必须分清：当前批 count、完整保存 E、最终输出 E。

```python
# 内核中，默认参数为0时与原来布局相同
z_stride = where(saved_edges > 0, saved_edges, n_edge)
z_row = z_ptr + (layer * F + focus) * z_stride * ROW + local_edge * ROW

output_stride = where(output_edges > 0, output_edges, n_edge)
out_row = out_ptr + focus * output_stride * ROW + local_edge * ROW
```

### 3.2 独立调度模块

文件：`deepmd/pt_expt/kernels/triton/sezm/so2_edge_cohort.py`。

- `supports_edge_cohort`：仅支持已测 Neo FP32/CUDA/连续布局、F2/Cf32/L3/2门控层/apply_alpha=True，其他回退。
- `cohort_input_backward_impl`：固定容量8192，尾批完整 mask，原 GEMM tile 参数不变；独立 saved/output stride；不再生成内核源码。
- `_cohort_input_backward`：运行时循环是 opaque custom_op，支持在线 torch.compile，**独立 PT2/C++ 部署未验收**。
- `_cohort_alpha_backward`：单独可见的 triton_op；使 x_local 的依赖只延续到 alpha 梯度完成，不必贯穿整个 input-gradient 循环。
- `cohort_backward`：组合两条输出；输入梯度 custom_op 不接收不使用的 x_local/u_final。

opt-in：在启动进程、导入模块及编译前设置 `DP_SO2_EDGE_COHORT_INFER=1`；默认0。切换需要新进程/新编译图。严格 FP32 仍设置 `DP_TRITON_INFER=2`、TF32/AMP关闭。

### 3.3 为什么内部 inplace 是安全的

这包含此前 #6058 的思想：只复用自己分配的 residual g。GEMM 的矩阵输入 gz 与输出必须分离；每个输出 tile 在覆盖 g 对应位置前，已读完自身 residual。其他 tile 不读取该 tile 的 residual。不是允许把任意 GEMM 写回其矩阵输入。

调用者输入、前向 z、alpha、权重均不修改，因此 `mutates_args=()` 仍符合实际。此已知优化与 ECS 收益重叠，不能相加。

## 4. 显存为何能降，为什么不等于“峰值按边批比例下降”

原生一阶分支的 g_cur / g_spare / gz 三块行张量与 sig，其分配规模可概括为：

```text
F * E * (3*ROW + gate_width) * sizeof(float)
```

这只统计这几块张量，不是整模型显存，也不是每一时刻的精确峰值公式。

ECS 保留一个完整最终梯度，再用有界 g/gz/sig：

```text
完整输出：F * E * ROW * 4
边批工作区：F * C * (2*ROW + gate_width) * 4
```

C8192时工作区48,234,496B，约46MiB；它仍是设备 global allocation，**不是片上shared，也不是保证驻留L2**。相较E647170，它不再随完整E线性增长，但完整输出和前向保存z仍增长。

整模峰值是所有存活张量占用曲线的最高点。某阶段减少3GB，不代表最高点一定下降3GB；峰值可能转移到另一个阶段。这也解释了相对已有inplace，ECS额外完整峰值只下降约2.06%，而不是把全部中间工作区差额兑现成整模下降。

## 5. 历史严格 FP32 实测

### 5.1 单个 stack 反向，不是 F+B 或整模

RTX5090同卡、同进程 CUDA Event，至少3次预热/10次正式采样，中位值；grad_u0/grad_alpha 对原生逐位一致。

| E | 原生反向 ms | ECS C8192 ms | 时间下降 |
| ---: | ---: | ---: | ---: |
| 158002 | 4.995616 | 3.968848 | 20.55% |
| 647170 | 19.537632 | 16.782304 | 14.10% |

### 5.2 完整模型：包含邻居构建的正常 ASE API

历史环境：Torch2.11.0+cu128、Triton3.6.0、driver595.91.07，RTX5090，严格FP32，无锁频。base=`668b60cc`。8次预热、30次正式墙钟采样，独立内存回放，首次执行计入全窗口峰值但不计稳态时间。GB=10^9B。

N1000每臂两个进程，表中时间为进程P50的中位数，峰值为最大值；其余GPU0每臂一次，N64另在GPU1反序重复。不是全部尺寸都已3进程验收。

| N | 原生→ECS P50 ms | 时间下降 | 吞吐提高 | 全窗口峰值 GB | 峰值下降 | 精度状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 18.448455→15.814499 | 14.277% | 16.655% | 0.498200→0.451064 | 9.461% | 对BASE和纯Torch三帧通过 |
| 512 | 23.799488→22.766906 | 4.339% | 4.535% | 3.116353→2.799795 | 10.158% | 对BASE通过；对纯Torch未全过 |
| 1000 | 40.872067→39.442227 | 3.498% | 3.625% | 5.968265→5.449051 | 8.700% | 两臂两进程对纯Torch通过 |
| 4096 | 156.495125→152.996582 | 2.236% | 2.287% | 24.061593→22.256657 | 7.501% | 对BASE第三帧virial失败，纯Torch OOM |

速度比例定义：时间下降=(T_base-T_new)/T_base；吞吐提高=T_base/T_new-1，两者不相同。N4096不足3%的速度变化只记观察值，不宣称稳定加速成功。

N64 GPU1反序：17.440020→15.244913ms，时间下降12.587%，峰值仍0.498200→0.451064GB。不同GPU绝对时间不混算。

### 5.3 与已知 SO2 inplace 对照，防止重复计算贡献

| N1000策略 | P50 ms | P90 ms | 全窗口峰值 GB | 稳态峰值 GB |
| --- | ---: | ---: | ---: | ---: |
| 已有B/inplace | 40.954964 | 41.079092 | 5.563780 | 5.464760 |
| B+ECS | 39.560225 | 39.582951 | 5.449051 | 5.449051 |

本轮相对B：时间下降3.406%，吞吐提高3.526%，全窗口峰值下降2.062%（114.729MB），稳态仅下降约0.29%。该增量每臂一次进程，证据强度弱于N1000裸基线的两进程表，需重复。

### 5.4 不能省略的失败

- C2048：E647170反向约49.319ms，比原生19.538ms慢很多。批量越小并非越快。
- N512：原生和候选相对纯Torch均存在近零virial严格失败；不能因为原生失败便判候选通过。
- N4096：候选对BASE第三帧virial误差/容差比1.394，严格失败；独立纯Torch参考实际OOM。不能拿N1000通过替代。
- 第二个water小checkpoint实际回退，新核调用0，只验证回退，不算多模型加速。
- 内核局部逐位一致，不代表完整图必然逐位一致；不归咎于“只是浮点波动”而跳过诊断。

验收仍为全元素 `abs(a-ref)<=1e-5+1e-4*abs(ref)`，finite，max_relative分母max(abs(ref),1e-8)。会议要求的纯Torch自比噪声与2～3倍经验区间另列，不能偷偷改严格阈值。

## 6. 算子封装曾使显存反而增加

第一版把所有 backward 输入塞进一个opaque custom_op，包括input-grad实际不用的u_final和只在grad_alpha需要的x_local。这让所有参数保活到长循环结束。N512虽局部暂存减少，整模峰值却从3.116GB增到3.317GB。

之后拆出可见alpha kernel，input-grad签名不再接收这两个张量。**相同GPU1、相同C16384**控制：3.316562→2.799795GB，峰值下降15.581%；时间22.670595→22.709203ms，基本不变。

这个15.581%是修复我们初版不良封装造成的回归，不是可叠加到最终8.70%的额外收益。原生可见triton_op本来就能去掉某些无用依赖，不能把原生已有能力说成新发现。

## 7. 为什么 torch.compile 默认没有自动完成 ECS

### 7.1 不是“PyTorch不懂SO(2)”这么简单

编译器并不需要理解完整物理含义才可能做循环交换。关键是能否证明：这一维独立、张量哪些区间被读写、层间别名合法、分批不改变归约语义，以及如何保留高效现有内核。

### 7.2 需要区分三类优化

| 优化 | 改变什么 | 是否等于ECS |
| --- | --- | --- |
| buffer reuse / inplace | 用完的空间重新使用 | 否，张量仍可能覆盖全E |
| peak-memory reorder | 调整整张量的申请/计算先后 | 否，不自动把其缩成C |
| cohort streaming | 改变跨层/跨边批循环嵌套与存储extent | 是本例核心 |

“编译器已优化allocation”不等于“已搜索这个跨核调度空间”。ECS 不只是把 `empty` 向后移。

### 7.3 原生 triton_op 并非完全黑盒

本机Torch2.11源码 `torch/_library/triton.py::triton_op` 明确区分：custom_op是opaque，triton_op允许compile/export看到其实现。原base用triton_op，所以不能说Inductor完全看不到内部kernel。

但是看见调用、grid、输入输出，并不等于能从用户Triton程序中自动推导所有跨边独立性与区间依赖，更不等于能自行改写任意kernel的完整E stride、tail mask、GEMM grid、最后输出布局并选择C。把多个已有kernel改为一批批执行，需要比常规逐点融合/整张量复用更强的调度转换。

新实现反而有意将动态循环放进opaque op；因此特别缩减签名，避免人为增加保活。它有部署和优化可见性的代价，不能宣称“opaque更先进”。

### 7.4 性能选择也不是显然的

保留全E GEMM通常有更大并行度和更少launch；分批增加启动开销，却可能改善局部性。太小C退化，太大C收益消失。需要成本模型/实测选择，不能要求通用编译器把所有边独立算子一律分块。

本例证明的是：**在固定版本、固定用户kernel和当前调度中，存在人工利用领域依赖信息获得的优化机会。**尚未系统比较新版本PyTorch、替代表示或编译选项，不能证明所有PyTorch版本普遍缺少该能力。

## 8. 其他哪里可能用同样方法

| 候选 | 为什么可能成立 | 主要限制/先检查什么 |
| --- | --- | --- |
| 逐边MLP/径向网络的多层反向 | 边为独立行，类似row-separable链 | 原Inductor可能已融合；权重梯度有跨边归约 |
| 其他宽度/层数的mixing | 相同独立轴，工作集可按字节预算选择 | 尚未验证，宽通道cuBLAS分支不能直接复用本实现 |
| 多层token-wise残差MLP | token内特征变换独立 | attention/跨token归一化是屏障，不可跨越 |
| 节点侧FFN | 节点内角度/通道计算可局部连续消费 | N远小于E，收益可能小，先排除已融合布局 |
| 源节点融合的有界通道/角度组 | 控制活跃输出，避免spill并增加任务数 | 这是邻居归约调度，不是原封不动的ECS |
| 独立梯度分支缩减custom-op签名 | 无用参数保活可提高全局峰值 | 原base已能消除的不能再当贡献；需wrapper证据 |

带跨行softmax/attention、图消息聚合、batch统计、训练权重梯度的链不满足简单独立性；要么作为分区边界，要么显式生成partial reduction并验证数值。随机数/dropout也要保持逐元素随机映射，不能默认分批等价。

## 9. 是否足够形成可发CCF-B级别的创新点

### 9.1 当前判断：有研究问题，但现有工程案例本身还不够

循环分块、循环交换、跨算子局部性、buffer reuse、流式反向都不是新概念。Welder等已有tile-graph研究；FlashAttention等已有避免大中间量和分块计算；TrimSO(2)已有等变模型数据流优化。单一DPA4形状的手写实现，整模约3.5%时间、且部分精度和部署未验收，不能直接包装成通用编译器突破。

CCF等级不是成果自动分档，更不能承诺录用。目前没有充分查新、方法泛化或跨平台证据证明可发表性。

### 9.2 更可能形成研究贡献的升级

可研究题目：**面向可分离反向计算链、保持原生内核的内存预算流式调度**。以下均是待验证研究设计，不是已完成创新声明。

1. **依赖/所有权语义契约**：描述独立轴、跨轴归约、读写区间、布局、保存状态和别名限制；自动判断哪些核可以跨层流式执行。不能只靠函数名白名单。
2. **不改算术的调度转换**：自动生成full-shape saved tensor与cohort scratch之间的stride适配，保留原生GEMM列并行与累加次序；对不支持的核自动回退。
3. **全局峰值感知成本模型**：同时计入C、launch数、实际时钟下的kernel时间、工作区、saved tensor保活、寄存器/缓存压力。优化全模型最高点，而非只最小化局部scratch。
4. **算子边界与生命周期协同设计**：根据真正消费者拆分输出依赖，证明并验证不会因opaque封装增加保活；与调度选择联合而不是两个独立手工补丁。
5. **可靠部署**：运行时动态E、C++/AOTI/PT2、CUDA Graph、内存安全和异常路径，不能依赖Python源码替换或假shape常量。

### 9.3 必须补齐的证据

- 多个真实DPA4尺寸、不同稀疏图/度数、多个行独立网络族，而非只一套权重换N。
- 至少两种GPU架构验证成本模型/启用规则，未知形状有稳健fallback。
- 三路精度及纯Torch自身重复噪声，全元素与局部FP64参考；训练扩展需权重梯度/二阶/多步优化器对照。
- 公平对照当前Torch compile/autotune/Graph、已知SO2 inplace、#6060、手工ECS、失败的全融合PBS及相关tile-graph方法。
- 消融自动识别、批量选择、ownership复用、边界拆分、launch开销；保留无收益/退化区域。
- 完整profile验证机制；至少全流程峰值、首次/稳态延迟、吞吐、compile成本，不能只给kernel最优值。

可作为停止/转向条件：若收益仅由已有inplace解释；只对一个固定形状有效；原生新版本已覆盖；通用方法不如手写规则且不能说明原因；或无法通过严格数值/部署验收，则应作为工程优化而不是强行包装论文。

## 10. 本次review提交的验证与限制

2026-10-19，独立fork工作区移植版：新CUDA测试 **45 passed**。包含36组输入梯度/alpha梯度逐位一致与输入不变检查（E=0/1/37/129/1025/8193；C64/8192；随机/零/小alpha），在线dynamic fullgraph compile，未支持布局守卫，训练/高阶路由守卫，以及冻结分支实际dispatch。守卫测试不是完整训练数值回归。

全仓 `ruff check .` 通过；`ruff format .` 检查2013个文件无额外格式变动。没有进行新native CUDA构建或全后端CI，没有声称45项测试等价于完整训练验收。

新移植版N64正常ASE API smoke已完成：生成wrapper包含两次`so2_cohort_input_bwd`调用，能量/力/位力/stress对历史同帧纯Torch参考通过，最大误差/容差比0.290686；三轮自身重复全部通过。全窗口allocated峰值451,063,808B，P50=13.123442ms、P90=13.198317ms，首次编译/调用约126.63秒。

本轮同GPU0原生对照随后完成：P50=12.172306ms、P90=12.250044ms，完整峰值498,199,552B；对相同历史纯Torch参考通过。每臂8预热/30样本，各一个独立进程，candidate先于base执行。移植版时间增加7.814%，峰值下降9.461%。**今日未复现小N加速，不将历史正向比例直接作为新移植版收益。** 此外，纯Torch参考是昨日保存的同帧输出，而非今日新跑的三路多进程噪声验收。

该差异需要反序重复并检查主机发射间隙、kernel实际时间/寻址生成代码；目前原因未定位。不把一般性的缓存假设当成已证实原因。本条是小规模执行/移植验证，不是多模型验收或所有N的性能复测。

未完成：多模型新增精度矩阵、N512/N4096严格问题定位、三进程正式性能复测、全perf组合、原生launcher/PT2/C++部署、跨架构验证。因此保持Draft、默认关闭，先供个人review。

## 11. 复现与证据索引

正常项目CUDA环境中，运行：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m pytest -q \
  source/tests/pt_expt/kernels/test_so2_edge_cohort.py
ruff check .
ruff format --check .
```

模型性能必须使用正常能量/力API，不使用会关闭力求导的inference_mode。两臂相同checkpoint、坐标、线程、allocator，各自新进程和cache；启用候选只增加 `DP_SO2_EDGE_COHORT_INFER=1`。完整协议与runner在用户实验目录，不提交权重、缓存或大数组到生产PR。

以下是服务器**本地证据路径，不是GitHub下载链接**：

```text
/home/kmpl/mkl/project/deepmd-kit/labs/dpa4_analysis/megakernel_20261018/
  REPORT.md / OUTCOME_TABLE.md / MEASURED_TABLE.md / summary.json
  results/cohort_scan158k/result.json
  results/cohort_scan647k/result.json
  results/model_n1000_{BASE,SPLIT}_r{0,1}/
  results/model_n1000_B_r0/ / results/model_n1000_B_SPLIT_r0/
  wrapper_audit.json / final_source_snapshot.json

/home/kmpl/mkl/project/deepmd-kit/labs/dpa4_analysis/ecs_pr_20261018/
  VALIDATION_PROTOCOL.md / validate_port.py / launch_port_model.py
  results/port_tests_r0/ / results/native_n64_r0/
```

## 12. 对导师可能追问的简短回答

**这和调GEMM tile一样吗？** 不一样：GEMM调度不变，改变跨层处理边的宏观顺序和中间张量范围。

**是不是把大张量放进shared了？** 不是，工作区仍是global allocation，缓存收益需要profile验证。

**为什么算得少了？** 公式和工作量基本没少，减少的是临时存储范围，改善数据消费局部性；同时增加了launch开销。

**数学等价是否保证整模逐位一样？** 不能。局部测试逐位相同，但完整图可能涉及其他归约/调度差异，N4096已有严格失败需保留。

**可否立即说最大原子数提高？** 不能，只测得特定规模峰值下降，最大容量与首次峰值、其他模块瓶颈相关，要实际探测。

**主要发现是什么？** 内核内部tile优化和整张量内存复用之外，还存在跨核、跨层、独立轴上的存储调度机会；但要同时尊重原生核并行度与图级生命周期。

## 参考资料

- [Megakernels](https://github.com/HazyResearch/Megakernels)：持久化/工作集管理的研究启发，不是本实现直接依赖。
- [PyTorch triton_op文档](https://docs.pytorch.org/docs/2.11/library.html#torch.library.triton_op)；本机对应源码 `torch/_library/triton.py` 已核对，不能把原生triton_op当完全opaque。
- [Welder, OSDI 2023](https://www.usenix.org/conference/osdi23/presentation/shi)：tile-graph与跨算子内存调度的相关工作，说明“分块”本身不新。
- [TrimSO(2)](https://arxiv.org/abs/2610.20029)：等变模型数据流融合的相关工作，需要明确区别与公平比较。
