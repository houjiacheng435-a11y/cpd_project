# Breakpoint Sampling Methods

这份说明对应 [extended_breaking_points.py](/C:/Users/houji/Desktop/cpd_project/breakpoint_sampling/extended_breaking_points.py) 里的 `FuncEvtCPD`。

这里所有方法都作用在 `dif` 上。`dif` 是增量序列，比如价格差分、收益、残差差分。  
这里的 `vol` 不是固定的理论分位数常数，而是外部给定的阈值序列或标量，用来直接控制输出事件数。  
因此，这里的统一判别逻辑是：

$$
\mathrm{stat}_t > \mathrm{vol}_t
$$

也就是说，方法本身负责构造统计量，`vol` 负责控制触发频率。

## 公共机制

- `b2b`：要求连续 `b2b` 次超阈值才记一次事件。
- `_prep`：清洗 `dif`，并把 `vol` 对齐到 `dif.index`。
- `_times`：把内部整数位置转换成时间戳。

## swt

Sliding Window Test。  
思想是把当前点 $x_t$ 和过去一个滚动窗口的均值 $\mu_t$ 比较：

$$
\mu_t = \mathrm{mean}(x_{t-\mathrm{window}}, \ldots, x_{t-1})
$$

统计量是：

$$
\mathrm{stat}_t = |x_t - \mu_t|
$$

如果当前增量相对最近窗口均值偏离很大，就触发事件。  
它本质上是最简单的局部均值偏移检测。

适合：

- 想看“当前点是否突然偏离最近常态”
- 想要短记忆、快响应

局限：

- 对噪声敏感
- 只看单点对局部均值的偏离，不累计证据

## cum

经典双边 CUSUM 的简化版。  
分别累计向上和向下的偏移：

$$
C_t^+ = \max(0, C_{t-1}^+ + x_t - \mathrm{drift})
$$

$$
C_t^- = \max(0, C_{t-1}^- - x_t - \mathrm{drift})
$$

统计量是：

$$
\mathrm{stat}_t = \max(C_t^+, C_t^-)
$$

只要持续有同方向偏移，统计量就会积累；若没有持续偏移，就回落到 0。  
所以它不是看单点异常，而是看“连续小偏移是否累计成结构性变化”。

适合：

- 持续性均值漂移
- 想检测弱但连续的变化

## page_hinkley

双向 Page-Hinkley。  
它维护一个在线均值：

$$
M_t = M_{t-1} + \frac{x_t - M_{t-1}}{t}
$$

然后分别累计向上和向下的偏离：

$$
\sigma_t^+ = \max\left(0, \alpha \sigma_{t-1}^+ + (x_t - M_t - \delta_{\mathrm{ph}})\right)
$$

$$
\sigma_t^- = \max\left(0, \alpha \sigma_{t-1}^- + (M_t - x_t - \delta_{\mathrm{ph}})\right)
$$

统计量是：

$$
\mathrm{stat}_t = \max(\sigma_t^+, \sigma_t^-)
$$

当当前值持续高于均值时，$\sigma^+$ 增长；当当前值持续低于均值时，$\sigma^-$ 增长。  
适合双向变点检测。

## cusum_ls

Local-mean standardized CUSUM 的简化版。  
和 `cum` 的区别在于，不直接累计原始 $x_t$，而是先减去一个局部基线均值：

$$
z_t = x_t - \mu_t
$$

其中 $\mu_t$ 来自最近 `warmup` 个点的滚动均值。然后再做双边累计：

$$
G_t^+ = \max(0, G_{t-1}^+ + z_t - \mathrm{drift})
$$

$$
G_t^- = \min(0, G_{t-1}^- + z_t + \mathrm{drift})
$$

统计量是：

$$
\mathrm{stat}_t = \max(G_t^+, |G_t^-|)
$$

这比 `cum` 多了一层“去局部均值”的处理，更像在检测相对当前局部状态的偏移。

适合：

- 序列均值本身会慢慢漂移
- 想看相对局部常态的变化

## sprt

Sequential Probability Ratio Test 的简化累计版本。  
这里没有显式构造完整似然比，而是用去局部均值后的残差做线性累加：

$$
z_t = x_t - \mu_t
$$

$$
G_t = G_{t-1} + z_t - \mathrm{drift}
$$

统计量是：

$$
\mathrm{stat}_t = |G_t|
$$

它和 `cusum_ls` 很像，但没有正负两侧分别做“截断回零”，所以更接近单条累计轨迹。

适合：

- 想保留顺序累计信息
- 想看单一累积证据是否越来越大

局限：

- 没有双边 reset 结构时，更容易受长时漂移影响

## gma

Geometric Moving Average。  
本质是对“相对局部均值的偏移”做指数加权平滑：

$$
g_t = \lambda g_{t-1} + (1 - \lambda)(x_t - \mu_t)
$$

统计量是：

$$
\mathrm{stat}_t = |g_t|
$$

$\lambda$ 越大，记忆越长；越小，越看重最新点。  
它可以看成 EWMA 型检测器的简化实现。

适合：

- 想抑制噪声
- 想比单点法更稳定、比 CUSUM 更平滑

## glr

Generalized Likelihood Ratio 的窗口版。  
在每个时点 $t$，枚举一个候选切分点 $k$，把区间分成左段和右段，比较两段均值是否显著不同：

$$
\mathrm{stat}(k, t) =
\frac{(\mathrm{mean}_{\mathrm{left}} - \mathrm{mean}_{\mathrm{right}})^2}
{\mathrm{var}_{\mathrm{left}}\left(\frac{1}{n_{\mathrm{left}}} + \frac{1}{n_{\mathrm{right}}}\right)}
$$

最后取当前时点下所有候选切分里的最大值：

$$
\mathrm{stat}_t = \max_k \mathrm{stat}(k, t)
$$

这是典型的“当前窗口内是否存在一个最优断点”思路。

适合：

- 明确想检验窗口内均值是否发生分段变化
- 希望检测更接近“一个真实切点”

局限：

- 计算量比前面累计类方法大
- 对窗口长度、左右最小样本数更敏感

## brandt_glr

Brandt 风格的 GLR。  
思想是用一个短窗口作为“当前状态”，用更早的历史作为“背景状态”，比较两者均值是否显著不同：

$$
\mathrm{stat}_t =
\mathrm{window} \cdot \frac{(\mathrm{mean}_{\mathrm{window}} - \mathrm{mean}_{\mathrm{global}})^2}{\mathrm{var}_{\mathrm{global}}}
$$

其中：

- `mean_window`：最近 `window` 个点均值
- `mean_global`：更早历史的全局均值
- `var_global`：更早历史的全局方差

和 `glr` 不同，它不在窗口内搜索最优切分点，而是直接比较“最近窗口”与“历史背景”。

适合：

- 关注最近局部状态是否明显偏离长期背景
- 想做在线背景对比

## e_detector

E-process / betting detector 风格的方法。  
先把数据标准化成：

$$
z_t = \frac{x_t}{\mathrm{std}_t}
$$

然后对一组 $\lambda$ 构造资本过程：

$$
E_t(\lambda) = \exp\left(\lambda z_t - \frac{1}{2}\lambda^2\right)
$$

代码里进一步把资本过程递推成：

$$
K_t(\lambda) = E_t(\lambda)\max(K_{t-1}(\lambda), 1)
$$

最后把多个 $\lambda$ 的资本做平均：

$$
\mathrm{stat}_t = \mathrm{mean}_{\lambda} K_t(\lambda)
$$

如果数据长期偏离零均值假设，资本会快速增长。  
这是更偏检验论、在线显著性监控的一类方法。

适合：

- 想用一组方向/尺度下注来覆盖不同变化强度
- 想要较强的在线检验解释

## ssr_cusum

Signed Sequential Rank CUSUM。  
先取 $|x_t|$ 在当前段内的秩，再乘以符号 $\mathrm{sign}(x_t)$，构造有符号秩分数：

$$
\xi_t = \mathrm{sign}(x_t) \cdot \mathrm{rank}(|x_t|) \cdot
\sqrt{\frac{6}{(2n+1)(n+1)}}
$$

再做双边 CUSUM：

$$
D_t^+ = \max(0, D_{t-1}^+ + \xi_t - \zeta)
$$

$$
D_t^- = \min(0, D_{t-1}^- + \xi_t + \zeta)
$$

统计量是：

$$
\mathrm{stat}_t = \max(D_t^+, |D_t^-|)
$$

它的核心是秩而不是原始幅度，所以相对更稳健，受重尾和极端值影响更小。

适合：

- 数据重尾
- 不希望少数极端值完全主导检测

## adaptive_cusum

自适应 CUSUM。  
它不是只盯均值变化，而是同时考虑：

- 均值上升 / 下降 / 不变
- 方差上升 / 下降 / 不变

所以代码里为多个方向组合维护并行统计量。  
每个方向 $d$ 上，都会构造一个对数似然比增量 $\mathrm{llr}_t(d)$，然后做：

$$
C_t(d) = \max(0, C_{t-1}(d) + \mathrm{llr}_t(d))
$$

最终统计量是所有方向里的最大值：

$$
\mathrm{stat}_t = \max_d C_t(d)
$$

与此同时，`mu_hat(d)` 和 `theta_hat(d)` 会随着检测过程更新，所以它是“边检测边估计”的。

适合：

- 不只关心均值变化，也关心波动结构变化
- 需要一个多备择、自适应的在线方法

局限：

- 参数较多
- 行为比简单 CUSUM 更复杂，更依赖调参

## aff

来源：*Continuous monitoring for changepoints in data streams using adaptive estimation*。

Adaptive Forgetting Factor。  
它来自自适应遗忘均值估计的思路。先维护两个递推量：

$$
m_t = \lambda_t m_{t-1} + x_t
$$

$$
w_t = \lambda_t w_{t-1} + 1
$$

于是自适应遗忘均值为：

$$
\mathrm{mean}_t = \frac{m_t}{w_t}
$$

和固定 $\lambda$ 的 EWMA 不同，$\lambda_t$ 不是常数，而是按预测误差梯度去更新。  
代码里先计算均值对 $\lambda$ 的导数，再用平方误差梯度更新：

$$
\mathrm{grad}_t =
2(\mathrm{mean}_{t-1} - x_t)\frac{d\,\mathrm{mean}_{t-1}}{d\lambda}
$$

$$
\lambda_t = \lambda_{t-1} - \mathrm{step} \cdot \frac{\mathrm{grad}_t}{\mathrm{var\_scale}}
$$

最后把 AFF 均值和一个初始基线均值比较：

$$
\mathrm{stat}_t = |\mathrm{mean}_t - \mathrm{baseline}|
$$

如果偏离足够大，就触发事件。

它的核心不是“固定记忆长度”，而是根据数据自己调整遗忘速度：

- 平稳时，倾向保留更多历史
- 变化时，倾向更快忘掉旧数据

## 方法之间的大致区别

- `swt`：单点对局部均值的偏离
- `cum`：原始增量的双边累计
- `page_hinkley`：在线均值下的双向偏离累计
- `cusum_ls`：去局部均值后的双边累计
- `sprt`：去局部均值后的单轨累计
- `gma`：去局部均值后的指数平滑累计
- `glr`：窗口内搜索最优切分点
- `brandt_glr`：最近窗口对长期背景
- `e_detector`：资本过程 / e-process
- `ssr_cusum`：有符号秩的稳健 CUSUM
- `adaptive_cusum`：均值和方差联合自适应
- `aff`：遗忘因子自适应的均值跟踪

## 和标准教材写法的差异

这份代码不是在复现每个方法的标准显著性阈值公式，而是在统一一个实验接口：

- 输入统一为 `dif`
- 阈值统一为 `vol`
- 输出统一为事件时间戳

所以这里更像“同一套事件采样框架下的不同统计量生成器”，而不是逐篇论文的原始检验器一比一复刻版。
