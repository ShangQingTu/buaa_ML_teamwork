# 1.团队作业中分工和贡献

## 1.1 分工

- 同学A负责搭建比赛的pipeline，在服务器上进行了训练和调参
- 同学B负责调研参考文献，撰写团队作业报告
- 同学C负责实现语言模型的代码

## 1.2 贡献

- A: 33%
- B: 33%
- C: 33%

# 2.模型原理

Liu等人将近代NLP技术的发展总结为四种范式[^1]，分别为：

- P1. 非神经网络时代的完全监督学习 （Fully Supervised Learning, Non-Neural Network）

- P2. 基于神经网络的完全监督学习 (Fully Supervised Learning, Neural Network)

- P3. 预训练，精调范式 (Pre-train, Fine-tune)

- P4. 预训练，提示，预测范式（Pre-train, Prompt, Predict）

我们对第2、第3和第4范式都实现了一个简单模型，进行了实验。

# 3.模型结构

## 3.1 Bert + TextCNN

现在实现了这个，这是范式3的做法

## 3.2 词向量+LSTM

这个是范式2的

## 3.3 Prompt in QA style

范式4

# 4.调参训练过程

同学A的服务器的配置是一张RTX 3090GPU， CUDA的版本是11.1，操作系统是Linux。

需要的python环境包在目录下的`requirements.txt`，可以用下面的命令配置:

```
pip install -r  requirements.txt
```

训练命令:

```
CUDA_VISIBLE_DEVICES=2 nohup python -m src.pipeline.train --batch_size 16 > bert_bs16.log &
```

测试:

```
CUDA_VISIBLE_DEVICES=2 nohup python -m src.pipeline.test --batch_size 16 --ckpt /home/tsq/buaa_ML_teamwork/ckpt/model_epoch4_val0.770.pt > bert_test_bs16.log &
```



# 5.结果



# 参考文献

[^1]: Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi,and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ArXiv,abs/2107.13586, 2021.

