# Hankel_DMD

这个目录现在只保留两部分内容：

1. 当前已经保留并可运行的 `CPD -> observable -> Hankel-DMD -> diagnostics` 主流水线。
2. 下一步打算转向的实时 DMD 状态建模思路。

历史实验脚本、PyDMD 对照、HH/LL 分块实验、map 型实验和对应结果目录都已经清掉，不再作为当前工作流的一部分。

## 当前保留的文件

```text
Hankel_DMD/
  configs/
    cpd_extrema_example.json
  scripts/
    build_cpd_extrema.py
    download_a_share_60min_akshare.py
    run_hankel_dmd.py
  __init__.py
  build_extremum_observable.py
  hankel_dmd.py
  diagnostics.py
  README.md
```

## 当前主流水线

### 1. 先跑 CPD 和极值确认

入口脚本：

```text
Hankel_DMD/scripts/build_cpd_extrema.py
```

示例配置：

```text
Hankel_DMD/configs/cpd_extrema_example.json
```

示例命令：

```powershell
python Hankel_DMD/scripts/build_cpd_extrema.py --config Hankel_DMD/configs/cpd_extrema_example.json
```

配置里的 `input_csv` 需要指向具体价格文件，例如：

```json
"input_csv": "data/a_share_1d_akshare/symbols/000001.csv"
```

默认输出目录形如：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/
```

该目录的基础输出是：

- `config.json`
- `cpd_diagnostics.csv`
- `confirmed_extrema.csv`

其中：

- `cpd_diagnostics.csv` 是逐 bar 的 CPD 诊断表；
- `confirmed_extrema.csv` 是后续 observable 和 DMD 的主要输入。

### 2. 从 confirmed extrema 构造 observable

核心函数：

```python
from Hankel_DMD import build_extremum_observable
```

示例：

```python
from pathlib import Path
from Hankel_DMD import build_extremum_observable

run_dir = Path("Hankel_DMD/outputs/runs/example_000001_cpd_extrema")

observable = build_extremum_observable(
    extrema_data=run_dir / "confirmed_extrema.csv",
    price_data=run_dir / "cpd_diagnostics.csv",
    symbol_col="stock_id",
    date_col="date",
    close_col="Close",
    price_mode="log",
)

observable.to_csv(run_dir / "observable.csv", index=False, encoding="utf-8-sig")
```

输出：

```text
Hankel_DMD/outputs/runs/<run_name>/observable.csv
```

当前 `observable.csv` 的关键列是：

- `event_id`
- `date`
- `bar_index`
- `extreme_type`
- `extreme_price`
- `observable`

### 3. 对 observable 跑 Hankel-DMD

入口脚本：

```text
Hankel_DMD/scripts/run_hankel_dmd.py
```

示例命令：

```powershell
python Hankel_DMD/scripts/run_hankel_dmd.py `
  --run-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema `
  --m 20 `
  --n 80 `
  --rank 5
```

参数含义：

- `m`：Hankel 矩阵行数；
- `n`：Hankel 矩阵列数减一，实际列数是 `n + 1`；
- `rank`：SVD 截断秩。

约束：

```text
m + n + 1 <= len(observable)
```

当前实现的一个重要细节：

- 不是拿整条 observable 一次性做全样本 Hankel；
- 而是只取 **最后 `m + n + 1` 个点**；
- 也就是 `run_hankel_dmd.py` 当前固定使用尾部窗口：

```text
used_window = tail_m_plus_n_plus_1
```

默认输出目录：

```text
Hankel_DMD/outputs/runs/<run_name>/hankel_dmd_m{m}_n{n}_r{rank}/
```

例如：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m20_n80_r5/
```

输出文件包括：

- `config.json`
- `eigenvalues.csv`
- `singular_values.csv`
- `modes_real.csv`
- `modes_imag.csv`

如果加上：

```powershell
--save-matrices
```

还会额外保存：

- `X.csv`
- `Y.csv`
- `A_hat_real.csv`
- `A_hat_imag.csv`

### 4. 跑最小诊断

入口：

```text
Hankel_DMD/diagnostics.py
```

只基于已有 DMD 输出做奇异值和特征值诊断：

```powershell
python Hankel_DMD/diagnostics.py `
  --dmd-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m20_n80_r5
```

如果还要计算基于当前参数的滚动窗口外下一点预测误差，需要补充：

```powershell
python Hankel_DMD/diagnostics.py `
  --dmd-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m20_n80_r5 `
  --observable-path Hankel_DMD/outputs/runs/example_000001_cpd_extrema/observable.csv `
  --m 20 `
  --n 80 `
  --rank 5
```

默认输出目录：

```text
Hankel_DMD/outputs/runs/<run_name>/hankel_dmd_m{m}_n{n}_r{rank}/diagnostics/
```

输出文件包括：

- `singular_value_diagnostics.csv`
- `eigenvalue_diagnostics.csv`
- `reconstruction_error.csv`

## 当前结果目录约定

现在只保留基础 run 目录，派生实验结果不再保留。

统一查看：

```text
Hankel_DMD/outputs/runs/<run_name>/
```

目前目录里通常只有：

- `config.json`
- `cpd_diagnostics.csv`
- `confirmed_extrema.csv`
- `observable.csv`

以及你手动再跑出来的：

- `hankel_dmd_m{m}_n{n}_r{rank}/`

## 当前实现和下一步方向要分开看

这个目录里现在真正已经实现并保留的，是：

- extremum observable 的构造；
- batch 式 Hankel-DMD；
- 最小诊断。

下面这部分不是当前已经落地的主实现，而是下一步准备推进的 **实时 DMD 状态建模设计**。

---

## 下一步 DMD 方向：实时状态 + DMDc

### 目标

在新的设计里，目标不再是“只对极值事件序列做一次性 DMD 分解”，而是构造一个：

- 以当前价格为基础；
- 带最近交替极值记忆；
- 可按 bar 实时更新；
- 仍然以 DMD 为核心的低维状态。

当前约束是：

- 必须用 DMD；
- 先不建模极值之间的完整路径；
- 只用：
  - 当前价格；
  - 最近 10 个交替确认极值。

### 为什么不继续沿用 extremum-only Hankel-DMD

原来的 event-time Hankel-DMD 路线有两个结构问题：

1. 只有确认出新极值时才更新，不是天然的 bar 级实时模型。
2. 如果把 high 和 low 混在一起，主结构很容易被“高低交替”本身占据，而不是更有意义的市场结构。

所以如果要做实时低维状态，更合理的是：

- 回到 bar-time；
- 让极值只承担“结构记忆”的角色。

### 建议的方法

建议使用：

**bar-time DMD with control (DMDc)**。

模型形式：

```text
x_{t+1} ~= A x_t + B u_t
```

含义：

- `x_t`：当前结构状态；
- `u_t`：下一步是否发生极值结构切换的事件输入；
- `A`：在没有结构切换时，状态如何自行演化；
- `B`：一旦确认出新极值，状态如何被改写。

相比 plain DMD，这样做的理由是：

状态更新其实有两种机制：

1. 普通 bar-to-bar 价格推进；
2. 新极值确认后，最近 swing 记忆整体右移和重写。

如果不用 `u_t`，这两种机制会被硬压到同一个线性算子里。

### 状态 `x_t` 的定义

设：

- `P_t`：当前 bar 价格；
- `E_k`：最近一个确认极值；
- `E_{k-1}, ..., E_{k-9}`：之前 9 个交替确认极值。

定义：

```text
x_t = [
  log(P_t) - log(E_k),
  log(E_k) - log(E_{k-1}),
  log(E_{k-1}) - log(E_{k-2}),
  ...,
  log(E_{k-8}) - log(E_{k-9})
]^T
```

这是一个 10 维状态：

1. 第一维：当前价格相对最近确认极值的偏移；
2. 后面 9 维：最近 9 段已完成的交替 swing 幅度。

这套定义的好处是很直接：

- 第一维回答“当前价格离最近结构锚点有多远”；
- 后面几维回答“最近这套 swing scaffold 长什么样”。

### 控制输入 `u_t` 的定义

最简单的一组输入是：

```text
u_t = [
  1(new extremum is confirmed at t+1),
  1(new high is confirmed at t+1)
]^T
```

典型情况：

- 没有新极值：

```text
u_t = [0, 0]^T
```

- 新 high 被确认：

```text
u_t = [1, 1]^T
```

- 新 low 被确认：

```text
u_t = [1, 0]^T
```

### DMDc 是怎么学的

收集一串三元组：

```text
(x_1, u_1, x_2), (x_2, u_2, x_3), ..., (x_{T-1}, u_{T-1}, x_T)
```

构造：

```text
X  = [x_1, x_2, ..., x_{T-1}]
Xp = [x_2, x_3, ..., x_T]
U  = [u_1, u_2, ..., u_{T-1}]
```

求解：

```text
Xp ~= A X + B U
```

也就是：

```text
Xp ~= [A  B] [X; U]
```

最小二乘估计为：

```text
[A  B] = Xp [X; U]^dagger
```

解释：

- `A` 学“结构不变时，状态自己怎么走”；
- `B` 学“新 extremum 事件会把状态怎么重写”。

### 一个最小例子

假设最近三个交替确认极值是：

- `E_{k-2} = 95`
- `E_{k-1} = 105`
- `E_k = 98`

当前价格：

- `P_t = 101`

缩成 3 维记忆时：

```text
x_t = [
  log(101 / 98),
  log(98 / 105),
  log(105 / 95)
]^T
```

如果下一根 bar 到 `102`，但没有新极值确认：

```text
u_t = [0, 0]^T
```

那么主要变化只在第一维。

如果下一步确认了一个新 high，例如 `106`：

```text
u_t = [1, 1]^T
```

那整个状态都会改写，因为：

- 当前偏移要相对新的锚点重算；
- 最近 swing 幅度记忆会整体右移；
- 插入一段新的已完成 swing。

这就是为什么需要 `B`。

### 两种“窗口”要分开

这里有两个不同概念的窗口。

#### 1. 结构记忆窗口

这是 `x_t` 里保留多少个极值。

当前建议：

- 保留最近 10 个交替确认极值。

这不是时间窗口。

#### 2. 模型拟合窗口

这是每次估计局部 DMDc 模型时，用多少根最近的 bar。

建议起点：

- 先用最近 `180` 根 bar；
- 但要求这段里至少有大约 `10` 次极值切换；
- 如果不足，可以扩到 `250` 根 bar 左右。

理由：

- 太短：事件太少，`B` 不稳；
- 太长：不同 regime 被平均。

### 这种模型能支持什么结论

这套设计更适合：

- 状态表示；
- 局部结构分析；
- 滚动跟踪。

它可以回答：

1. 当前价格加最近交替 swing 记忆，是否存在明显低维结构；
2. 当前时刻在 DMD 模态坐标下处在什么位置；
3. 最近 swing 结构是偏持续、偏振荡，还是偏快速衰减；
4. 如果做滚动重估，当前局部结构是否在漂移。

### 它明确不回答什么

因为当前设计里故意不放“极值之间的完整路径”，所以它不适合回答：

1. 当前腿内部到底平滑还是粗糙；
2. intraleg 波动聚集；
3. 两个 extremum 之间的详细形状；
4. 下一次 extremum 的精确发生时点。

因此它应该被理解为：

**swing-structure state model**，而不是完整路径模型。

### 一个实际输出

如果拿到 DMD 模态矩阵 `Phi`，一个自然的低维状态输出是：

```text
a_t = Phi^dagger x_t
```

这个 `a_t` 才是候选的实时低维向量。它满足：

- 每根 bar 更新一次；
- 仍然 anchored 在“当前价格 + 最近 swing 记忆”上；
- 保持在 DMD 框架内部。

## 当前推荐结论

在现阶段，这个目录里的建议路线分成两条：

### 已实现并可直接使用

1. `build_cpd_extrema.py`
2. `build_extremum_observable.py`
3. `run_hankel_dmd.py`
4. `diagnostics.py`

这条线适合继续做：

- extremum observable 的离线 Hankel-DMD；
- 奇异值、特征值和最小误差诊断。

### 下一步准备实现

1. 用当前价格和最近 10 个交替 extremum 构造 `x_t`；
2. 用 extremum 确认事件构造 `u_t`；
3. 做 rolling bar-time DMDc；
4. 输出实时低维状态。

一句话概括：

**当前保留实现是 batch Hankel-DMD；下一步设计方向是 bar-time DMDc 状态模型。**
