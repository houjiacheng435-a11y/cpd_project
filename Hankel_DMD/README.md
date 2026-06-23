# Hankel_DMD 用法说明

本目录现在作为 automatic123 + Hankel-DMD 实验的统一工作区。之后从头跑数据、保存中间结果、继续做 observable 和 DMD，都放在这里。

## 目录结构

```text
Hankel_DMD/
  scripts/
    build_cpd_extrema.py          # 从原始日频 CSV 重新跑 CPD + automatic123 极值
  configs/
    cpd_extrema_example.json      # 示例配置
  outputs/
    runs/
      <run_name>/                 # 每次运行一个独立目录
  build_extremum_observable.py    # 从 confirmed_extrema 构造 single observable
  hankel_dmd.py                   # 对 observable 执行 Hankel-DMD 核心计算
  diagnostics.py                  # 对 Hankel-DMD 输出做最小诊断
```

## 1. 从头跑 CPD 和极值点

入口脚本：

```text
Hankel_DMD/scripts/build_cpd_extrema.py
```

如果需要先下载 A 股 60 分钟数据，入口脚本：

```text
Hankel_DMD/scripts/download_a_share_60min_akshare.py
```

示例命令：

```powershell
python Hankel_DMD/scripts/download_a_share_60min_akshare.py `
  --source sina `
  --symbols 000001,000002,000006,000008,000009 `
  --start "2020-01-01 09:30:00" `
  --end "2026-05-29 15:00:00" `
  --out-dir data/a_share_60min_akshare
```

输出：

| 文件 | 用途 |
|---|---|
| `data/a_share_60min_akshare/symbols/<symbol>.csv` | 单只股票 60 分钟 OHLC 数据 |

说明：`source=sina` 通常返回最近约 1970 根 60 分钟 K 线；`source=eastmoney` 在当前接口下通常只返回更短的近期窗口。
下载完成后，脚本只在终端打印下载状态和共同小时 bar 日历下的完整性检查，不额外生成汇总 CSV。

示例命令：

```powershell
python Hankel_DMD/scripts/build_cpd_extrema.py --config Hankel_DMD/configs/cpd_extrema_example.json
```

示例配置：

```text
Hankel_DMD/configs/cpd_extrema_example.json
```

配置中的 `input_csv` 必须指向一个具体 CSV 文件，例如：

```json
"input_csv": "data/a_share_1d_akshare/symbols/000001.csv"
```

默认输出目录：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/
```

该目录里会生成：

| 文件 | 用途 |
|---|---|
| `config.json` | 本次运行使用的方法和参数 |
| `cpd_diagnostics.csv` | 日频级别的 CPD/Direction/Status/极值确认诊断 |
| `confirmed_extrema.csv` | 极值事件表，后续 Hankel-DMD 主要读取这个 |

`confirmed_extrema.csv` 的主要列：

| 列名 | 含义 |
|---|---|
| `event_id` | 极值事件 ID |
| `stock_id` | 股票代码 |
| `extreme_type` | `high` 或 `low` |
| `extreme_idx` | 极值所在 bar 位置 |
| `extreme_date` | 极值实际发生日期 |
| `extreme_price` | 极值价格，口径由配置里的 `log_price` 决定 |
| `confirmed_idx` | 极值被确认时的 bar 位置 |
| `confirmed_date` | 极值被确认日期 |

## 2. 从极值点构造 observable

入口函数：

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

out_path = run_dir / "observable.csv"
observable.to_csv(out_path, index=False, encoding="utf-8-sig")
```

输出：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/observable.csv
```

`observable.csv` 只包含：

| 列名 | 含义 |
|---|---|
| `event_id` | 极值事件 ID |
| `date` | 极值事件日期 |
| `bar_index` | 极值事件在日频价格序列中的位置 |
| `extreme_type` | `high` 或 `low` |
| `extreme_price` | 极值价格 |
| `observable` | 给 Hankel-DMD 使用的一维序列 |

## 结果查找规则

以后不要再去 `market_state_vector_builder/outputs/` 里找新结果。

新的结果统一看这里：

```text
Hankel_DMD/outputs/runs/<run_name>/
```

每次重新跑数据都新建一个 `<run_name>`，不要覆盖旧结果。

## 3. 执行 Hankel-DMD 并保存结果

入口脚本：

```text
Hankel_DMD/scripts/run_hankel_dmd.py
```

输入：

- 一维 `observable` 序列。
- `m`：Hankel 矩阵行数。
- `n`：Hankel 矩阵列数减一，实际列数为 `n + 1`。
- `rank`：SVD 截断 rank。

要求：

```text
m + n + 1 <= len(observable)
```

示例：

```powershell
python Hankel_DMD/scripts/run_hankel_dmd.py `
  --run-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema `
  --m 20 `
  --n 80 `
  --rank 5
```

默认输出目录：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m20_n80_r5/
```

输出目录会自动带上参数名，格式为：

```text
hankel_dmd_m{m}_n{n}_r{rank}/
```

例如 `--m 10 --n 80 --rank 10` 会写到：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m10_n80_r10/
```

该目录会生成：

| 文件 | 含义 |
|---|---|
| `config.json` | 本次 Hankel-DMD 参数 |
| `eigenvalues.csv` | DMD 特征值，包含实部、虚部、模长和相位 |
| `singular_values.csv` | Hankel 矩阵 `X` 的奇异值 |
| `modes_real.csv` | projected modes 的实部 |
| `modes_imag.csv` | projected modes 的虚部 |

如果需要同时保存 `X`、`Y` 和 `A_hat`，添加：

```powershell
--save-matrices
```

## 4. 运行最小诊断

入口脚本：

```text
Hankel_DMD/diagnostics.py
```

只基于已有 DMD 输出做奇异值和特征值诊断：

```powershell
python Hankel_DMD/diagnostics.py `
  --dmd-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m20_n80_r5
```

如果还要计算滚动窗口外下一个 observable 点的预测误差，需要提供 `observable.csv` 和本次 DMD 参数：

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
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd_m20_n80_r5/diagnostics/
```

可能生成：

| 文件 | 含义 |
|---|---|
| `singular_value_diagnostics.csv` | 奇异值、归一化奇异值、能量占比和累计能量 |
| `eigenvalue_diagnostics.csv` | 特征值实部、虚部、模长和相位角 |
| `reconstruction_error.csv` | 窗口外下一个 observable 点的 MAE、RMSE、中位绝对误差、midpoint 相对绝对误差和最后一个预测误差 |

## 5. PyDMD HankelDMD 对照

入口脚本：

```text
Hankel_DMD/scripts/run_pydmd_hankel_grid.py
```

如果本地没有 PyDMD，先安装：

```powershell
python -m pip install pydmd
```

示例命令：

```powershell
python Hankel_DMD/scripts/run_pydmd_hankel_grid.py `
  --run-dir Hankel_DMD/outputs/runs/example_spx_1d_cpd_extrema `
  --rank 5 `
  --local-grid-path Hankel_DMD/outputs/runs/example_spx_1d_cpd_extrema/parameter_grid_tail_r5/grid_summary.csv
```

默认输出目录：

```text
Hankel_DMD/outputs/runs/example_spx_1d_cpd_extrema/pydmd_hankel_grid_r5/
```

可能生成：

| 文件 | 含义 |
|---|---|
| `pydmd_grid_summary.csv` | PyDMD HankelDMD 参数网格预测误差 |
| `pydmd_grid_leading_eigenvalues.csv` | PyDMD HankelDMD leading eigenvalues |
| `comparison_with_local.csv` | PyDMD 与本项目实现的误差指标对比 |
| `eigenvalue_comparison_with_local.csv` | PyDMD 与本项目实现的 leading eigenvalues 对比 |
