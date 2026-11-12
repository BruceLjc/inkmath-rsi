# InkMath RSI：本地数学笔迹识别研究原型

输入已经采集好的分笔坐标，输出数学结构和 LaTeX。本轮从公开数字笔迹重新训练轻量模型，没有调用商业识别 API、预训练 OCR 或大语言模型。支持范围是数学符号与表达式，函数名和变量属于数学记号；不处理英文句子、中文、坐标系和几何图形。

**这是约一小时迭代的研究版本，尚不具备产品准确率，也不能承诺任意输入在 1 秒内正确识别。** 最终实测见 [验收报告](docs/iteration-report.md)。

## 直接运行

```sh
cd ~/code/my/inkmath-rsi
./start.sh
```

窗口支持手写、撤销、清空、载入公开笔迹例子、识别和保存人工标注样本。例子是从验证集选出的演示样本，不能用来衡量泛化准确率。界面上的 LaTeX 预览受 Matplotlib mathtext 支持范围限制，原始 LaTeX 文本始终保留。原生窗口交互仍需按下方步骤人工验收，离屏渲染已验证。

现有 `.venv` 使用本机已安装的依赖。换机器时：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[demo]'
```

依赖：Python 3.11+、PyTorch、NumPy、Pillow、NetworkX；实验窗口另需 Matplotlib。模型与训练数据保存在当前机器本项目目录，未上传或发布。

CLI（每次启动包含 Python/PyTorch 导入成本）：

```sh
.venv/bin/python -m inkmath examples/integral.json
```

常驻服务（用于亚秒级热推理）：

```sh
.venv/bin/python -m inkmath.server --port 18765
curl http://127.0.0.1:18765/health
curl -H 'Content-Type: application/json' --data-binary @examples/integral.json http://127.0.0.1:18765/recognize
```

默认读取 `artifacts/selected.json` 的权重与结构参数。只绑定 localhost。请求结构：

```json
{"task_id":"student-001","revision":3,"strokes":[[[10,20],[20,30]],[[10,30],[20,20]]]}
```

一个数组表示一笔，点为 `[x,y]`，y 轴向下；也支持每笔 `{ "x":[...], "y":[...] }`。保留真实分笔及书写顺序，既不需要压力/倾角/光学信息，也不接收照片作为识别输入。`expected_latex` 等额外标注不进入推理。

返回 `latex`、`tree`、`symbols`（含笔画索引）、`timing_ms`、`within_1s`、`task_id`、`revision`。`candidate`/`uncertain` 都是待核对候选；分数尚未校准，不能把高分当正确保证。服务另报 `server_total_ms`，不含客户端传输和界面呈现。

## 场景边界

| 场景 | 目标例子 | 本轮情况 |
|---|---|---|
| 四则、括号、变量 | `y=2x+1` | 已训练，仍可能混淆 x/乘号、1/l |
| 分数与嵌套分数 | `\frac{x+1}{x^2+1}` | 已实现二维结构解析 |
| 上下标 | `x^2+a_n` | 已实现；归属易出错 |
| 根式 | `\sqrt{x+1}`、`\sqrt[3]{x}` | 已实现结构导出，根指数样本较少 |
| 函数 | `\sin x+\cos x`、`\log_2 x` | sin/cos/tan/log 有训练标签 |
| 极限 | `\lim_{x\rightarrow 0}\frac{\sin x}{x}` | 有训练与独立评测；样本偏少 |
| 积分 | `\int_0^1 x^2\,dx` | 有训练与独立评测，含上下限结构 |
| 求和 | `\sum_{n=1}^{\infty}\frac{1}{n^2}` | 有训练与独立评测，含上下限结构 |
| 导数记号 | `f'(x)`、`\frac{dy}{dx}` | 基础符号可组合，未做专项达标验收 |
| 连乘、重积分、偏导、ln/exp 专项 | `\prod_{i=1}^n a_i`、`\iint f\,dx\,dy` | 训练标签/覆盖不足，当前不承诺支持 |
| 分段、矩阵、多行推导、全文批量 | — | 本版不支持 |

## 识别方式与速度边界

有限的连续/空间近邻笔画候选 → 双通道局部线条 CNN＋轨迹分支 → beam 分组覆盖 → 小型关系网络 → 最大生成树 → LaTeX 导出。模型参数与实际速度记录在验收报告。

推理上限：128 笔、12,000 点、900 个候选，每个候选最多取 16 笔邻域上下文；HTTP JSON 最大 512 KiB。超过上限会拒绝，模型忙时立即返回 429，不把排队藏在延迟统计外。每笔单独候选始终保留，分组必须恰好覆盖全部输入。

1 秒指标针对模型已经加载并预热的常驻 CPU 单请求，包含输入校验、候选特征、分类、分组、布局与导出。冷启动、客户端网络、写完等待时间、显示渲染不在此统计内。压力测试只能证明所测样本和机器上的耗时，不能给 Python/桌面系统任意输入提供实时上界保证。训练使用完整邻域上下文，运行时截断至 16 笔，是首版速度折中。

## 本地迭代

`artifacts/v1.pt` 到 `v5.pt` 保留实验权重，`reports/` 保留真实训练/评测记录。`best.pt` 和 `selected.json` 是当前晋级结果。验证集选模型，预留测试只在最终冻结后运行一次。

```sh
# 只比较已有候选；完整验证集，自动选择并晋级
.venv/bin/python scripts/rsi.py --models v4,v5 --limit 0 --promote
# 有边界的再训练：从 v4 初始化双头模型，最多 3 轮；仅看验证集
.venv/bin/python scripts/rsi.py --models v4,v5 --train --epochs 8 --rounds 1 --limit 0 --promote
# 验证接口与边界
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/smoke_api.py
.venv/bin/python scripts/benchmark.py --name benchmark-local --repeat 8
```

这是可追溯的训练、评测、选优循环；没有声称模型能自行发明算法或自动生成可靠标签。窗口保存的 `captures/` 样本含候选与人工标签，默认不会被训练脚本自动消费。后续应先建立真实学生、设备、书写风格分离的人工标注集，再依据验证集错误补数据和改模型。

从数据重现的顺序：`download_data.py` → `prepare.py` → `train.py`（v1、v2）→ `train_relations.py` → `mine_negatives.py` → `prepare_trajectories.py` → `prepare_context.py` → `train_context.py`（v4，随后 `--dual --name v5`）→ `rsi.py`。完整命令与覆盖说明见 [重新训练](docs/training.md)。历史指标对应本机已记录的权重/配置，并非跨软件版本完全确定。

## 数据许可与人工验收

公开 CROHME 归档带 **CC BY-NC-SA 3.0** 与各子集版权/分发限制。本轮数据及权重仅供本地研究，商用需替换为自有或明确授权的数据重新训练。详见 [数据来源](docs/data-sources.md) 和 [原始声明](docs/CROHME-original-notices.txt)。自写代码不代表训练数据获得商用授权。

人工验收：打开 `./start.sh`，先载入例子并识别，再清空后亲手写 `x^2+1`、定积分、求和与极限；核对 LaTeX 内容及上下限归属，快速改写时核对旧 revision 不覆盖新结果；保存人工修正标签后检查 captures 文件。最后在实际学生终端重复，记录正确率、端到端延迟和失败样例。当前未完成真人/真机书写验收。
