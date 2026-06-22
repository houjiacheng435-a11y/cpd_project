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
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd/
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
  --dmd-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd
```

如果还要计算一步映射误差和样本长度稳定性，需要提供 `observable.csv` 和本次 DMD 参数：

```powershell
python Hankel_DMD/diagnostics.py `
  --dmd-dir Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd `
  --observable-path Hankel_DMD/outputs/runs/example_000001_cpd_extrema/observable.csv `
  --m 20 `
  --n 80 `
  --rank 5 `
  --sample-sizes 101,120,147
```

默认输出目录：

```text
Hankel_DMD/outputs/runs/example_000001_cpd_extrema/hankel_dmd/diagnostics/
```

可能生成：

| 文件 | 含义 |
|---|---|
| `singular_value_diagnostics.csv` | 奇异值、归一化奇异值、能量占比和累计能量 |
| `eigenvalue_diagnostics.csv` | 特征值实部、虚部、模长和相位角 |
| `reconstruction_error.csv` | 相对一步映射误差 |
| `sample_size_stability.csv` | 不同末端样本长度下的 leading eigenvalues |
