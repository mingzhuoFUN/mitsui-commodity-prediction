# MITSUI&CO. Commodity Prediction Challenge

本项目用于训练 MITSUI&CO. Commodity Prediction Challenge 的 424 个多期限收益与收益差目标。

项目以最初的竞赛 notebook 为模型蓝本，保留其 target pair 特征、LightGBM、Random Forest、XGBoost、stacking 和 Kaggle inference server 主线。工程代码对标签对齐、时间序列泄漏和 stacking 训练方式进行严格处理，但不改变这套核心模型思路。

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

忠实复现入口：

```text
notebooks/mitsui_original_model_reproduction_colab.ipynb
```

该版本保持最初上传 notebook 的目标重建、特征、三基础模型、XGBoost
元模型和推理结构，仅调整 Colab 认证、数据路径和线程数量。它用于复现原
notebook 的 Kaggle 线上结果，与工程化实验模型分开。

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
