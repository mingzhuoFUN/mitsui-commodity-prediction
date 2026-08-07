# MITSUI&CO. Commodity Prediction Challenge

本项目用于训练 MITSUI&CO. Commodity Prediction Challenge 的 424 个多期限收益与收益差目标。

项目采用 target pair 因果特征、LightGBM、Random Forest、XGBoost、时间序列 OOF stacking 和 Kaggle inference server，覆盖从训练、验证到顺序推理的完整流程。

> 当前状态：本地源码、单元测试和小规模训练已经验证。完整 Colab
> notebook 已提供，但在一次全新 Colab runtime 完整执行并保存日志之前，
> 不把项目标记为“完整 Colab 验证完成”。下述结果是本地时间留出分数，
> 不是 Kaggle public 或 private leaderboard 分数。

## 完整建模思路

```text
官方市场数据与标签对齐
        ↓
目标 pair 因果特征 + 延迟可用的历史标签特征
        ↓
带 embargo 的时间序列切分
        ↓
LightGBM / Random Forest / XGBoost
        ↓
时间序列 OOF 预测
        ↓
Ridge stacking 元模型
        ↓
424 目标预测与 Daily IC / IC Sharpe
        ↓
连续历史 Kaggle inference server
```

### 数据和标签

- `train.csv`：股票、外汇、LME 与 JPX 市场时间序列。
- `train_labels.csv`：正式训练使用的 424 个官方标签。
- `target_pairs.csv`：每个目标对应的预测期限和一个或两个资产。
- `label_lags_1` 至 `label_lags_4`：推理阶段刚刚变得可用的历史标签。

训练时 `X[t]` 与官方 `Y[t]` 按 `date_id` 对齐。价格重建标签只用于检查公式，不替代官方标签。

### 因果特征

每个目标使用其 pair 资产的：

- 当前对数价格；
- 1、2、3、5、10、20 日对数收益；
- 5、20、60 日滚动收益均值与波动率；
- 两资产对数价差、价差变化和滚动 z-score；
- 按目标 horizon 模拟延迟公开的历史标签。

所有 `feature[t]` 只依赖时间不晚于 `t` 的信息。不使用负数 shift、未来差分或 backward fill。

### 验证与 stacking

验证集固定在时间轴末尾，训练与验证之间设置 4 天 embargo。基础模型的 stacking 输入来自 `TimeSeriesSplit` 生成的 OOF 预测，元模型不会看到基础模型对自身训练样本的拟合预测。

最终比较：

- Ridge baseline；
- LightGBM；
- Random Forest；
- XGBoost；
- 三模型 OOF stacking。

## 已验证结果

| 模型 | 目标数 | 验证天数 | IC Sharpe | Mean Daily IC |
|---|---:|---:|---:|---:|
| Ridge baseline | 424 | 252 | 0.1466 | 0.0242 |
| LGBM + RF + XGB OOF stacking | 424 | 252 | **0.2005** | **0.0434** |

以上是本地时间留出结果，不是 Kaggle leaderboard 分数。

运行环境和结果口径：

- 训练使用官方 `train_labels.csv`，验证集为时间轴末尾 252 天；
- 训练集与验证集之间使用 4 天 embargo；
- 当前表格是仓库实现的 Daily IC / IC Sharpe，不代表 public 或 private LB；
- 随机种子在基础模型中固定为 42；
- 完整运行报告还需要记录 Colab Python、依赖版本、硬件和实际执行日志。

完整 Kaggle local gateway 已成功运行 134 个测试日，验证了：

- 连续市场历史缓存；
- `label_date_id` 历史标签更新；
- 每批 424 列预测及正确列顺序；
- competition inference server 调用流程。

## 项目结构

```text
notebooks/          # Colab/Kaggle 完整训练与提交入口
scripts/
  train_baseline.py # Ridge 基准
  train_ensemble.py # 三模型 OOF stacking
  run_local_gateway.py
src/mitsui/
  features.py       # 因果特征
  ensemble.py       # 基础模型、OOF 与 stacking
  inference.py      # 连续历史在线推理
  metric.py         # Daily IC 与 IC Sharpe
  validation.py     # 时间切分与 embargo
tests/              # 指标、泄漏和推理测试
```

主要运行入口：

```text
notebooks/mitsui_competition_colab.ipynb
```

该 notebook 会从 GitHub 安装项目、下载 Kaggle 数据、运行测试和冒烟训练、执行完整 424 目标验证、用全部数据训练提交模型，并初始化 competition inference server。

当前完整实现位于远端 `agent/full-competition-pipeline` 分支，Colab
入口会显式克隆该分支。分支合并到 `main` 后，应同步把 notebook 中的
`REPO_REF` 改为 `main`。

模型实验入口：

```text
notebooks/mitsui_model_experiments_colab.ipynb
```

该 notebook 提供生成目标、lag/rolling/difference 特征、三基础模型、
XGBoost 元模型和 inference server 的另一条实验路径。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
```

竞赛数据解压到 `data/raw/`。CSV、ZIP、模型、输出和凭证由 `.gitignore` 排除。

Colab 推荐将 Kaggle 新版 `KGAT_...` token 保存为名为
`KAGGLE_API_TOKEN` 的 Secret。Legacy `kaggle.json` 仅作为备用。

Colab 训练完成后，notebook 会将模型、指标和验证预测复制到
`MyDrive/mitsui-artifacts/` 并检查文件是否存在且非空。

### Colab 常见问题

- `Repository not found` 或找不到训练脚本：确认 `REPO_REF` 指向存在的远端分支；
- Kaggle 返回 401/403：确认已经加入比赛，并给 notebook 开启 Secret 访问；
- `No module named xgboost`：重新运行依赖安装单元；
- runtime 中断：从 Google Drive 恢复已经持久化的模型；完整训练目前仍需增加 target 级断点续训；
- 内存或时间不足：先运行 8-target smoke，确认成功后再执行完整 424-target 训练。

## 运行

测试：

```powershell
pytest
```

小规模检查：

```powershell
python scripts/train_ensemble.py `
  --data-dir data/raw `
  --output-dir outputs/ensemble_smoke `
  --valid-size 128 `
  --max-targets 8
```

完整时间验证：

```powershell
python scripts/train_ensemble.py `
  --data-dir data/raw `
  --output-dir outputs/ensemble_full `
  --valid-size 252
```

使用全部训练数据拟合提交模型：

```powershell
python scripts/train_ensemble.py `
  --data-dir data/raw `
  --output-dir outputs/ensemble_submit `
  --fit-full
```

本地运行 Kaggle gateway：

```powershell
python scripts/run_local_gateway.py `
  --data-dir data/raw `
  --model-path outputs/ensemble_submit/stacked_models.pkl
```

## 数据安全

不要提交：

- `train.csv`、`test.csv` 和其他竞赛数据；
- Kaggle ZIP；
- `kaggle.json`、token 或 `.env`；
- `.pkl` 模型和训练输出。
