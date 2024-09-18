### 1：数据处理

首先，运行 `data_process.py` 文件。此步骤会读取 `data/unlabeled_data.csv` 文件，并生成以下两个文件：

- `task2_dataset.csv`: 用于 BERT 任务的数据集
- `vocab.txt`: 用于训练的词汇表

```shell
python data_process.py
```

### 2：模型训练

接下来，运行 `train.py` 文件进行模型训练。训练过程中将自动记录以下内容，并保存在 `result` 文件夹下：

- 训练损失 (loss)
- MLM 任务的预测准确率
- NSP 任务的预测准确率

```shell
python train.py
```

### 3：结果可视化

最后，运行 `plot_result.py` 文件，将训练损失和两个任务的预测准确率绘制成图表，便于分析和评估模型表现。

```shell
python plot_result.py
```


生成的图表会显示训练过程中损失的下降趋势以及 MLM 和 NSP 任务的准确率变化。
