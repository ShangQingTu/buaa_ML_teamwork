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

现在实现了这个，这是范式3的做法。

Bert模型[^2]是一个预训练的语言模型，我们只是拿来微调了一下，并结合我们文本分类的任务，在模型的特征输出层之后，接了一个TextCNN层[^3]提取文本的特征。

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

- bert

在线下测试，采用的是训练集的2000条测试的，Bert的`F1`分数是0.77

Bert的在线测试结果是0.72，如下图:

![](./pic/bert_result.png)

- bi-lstm

bi-lstm训练20个epoch的线下准确率(accuracy)是0.92，在线测试`F1`分数是0.31

bi-lstm训练4个epoch的线下准确率(accuracy)是0.78，在线测试`F1`分数是0.32

说明bi-lstm非常容易过拟合

- prompt

prompt采用的模板是`{"placeholder":"text_a"} It was {"mask"}`

用一个词到标签的关系来把`mask`位置的预测词映射到情感标签上，采用的映射关系是:

```python
label_words = {
    "negative": ["bad"],
    "neutral": ["normal"],
    "positive": ["good", "wonderful", "great"],
}
```

由于我们只是设定了模板和映射关系，完全没有改变模型的参数，我们也完全没有使用训练集里的数据，所以这样是一个无监督的过程。获取的在线测试`F1`分数是0.3078，和baseline差不多。

总体来看，还是`预训练+微调`的范式能够在**有训练数据**的情感分类任务中取得最高的分数。

# 参考文献

[^1]: Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi,and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ArXiv, abs/2107.13586, 2021.

[^2]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers),pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

[^3]: Yoon Kim. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746–1751, Doha, Qatar,October 2014. Association for Computational Linguistics.