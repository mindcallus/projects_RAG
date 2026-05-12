# 脑电语义解析 Demo 项目说明

## 项目背景

脑电语义解析的核心问题，是在阅读或语言加工场景下，尽量从 EEG 信号中恢复与语义相关的信息。这个方向的价值不在于“直接读心”，而在于验证脑电中是否包含可被模型利用的语言线索，以及探索脑机接口、认知计算和神经语言建模之间的连接方式。

当前项目以公开阅读任务数据为基础，围绕两条主线展开：

1. 句子级 EEG-to-Text 解码：将与句子/词对齐的脑电特征映射为文本输出。
2. 语义表征 probing：冻结解码模型后，分析其中间表征是否保留了词频、词长、词性/内容词-功能词等语义或语言学属性。

这个 demo 更接近“研究型工程验证”而不是产品化系统，重点是把数据、训练、推理和结果导出链路跑通。

## 任务目标

本项目的任务目标可以概括为两层：

1. 工程层目标：把公开 EEG 阅读数据整理成可训练格式，完成模型训练、推理评测和结果落盘。
2. 研究层目标：观察 EEG 表征经过深度学习模型后，是否能支持文本解码，或至少保留部分可探测的语义相关结构。

在当前仓库里，主任务不是实时脑电采集，而是基于离线数据做解码与分析。

## 数据与预处理

### 1. 数据输入形式

当前代码主要消费的是已经整理好的 pickle 数据，而不是原始采集设备流。

- 数据样例路径：
  - `data/zuco2/pickle/zuco2_source.pkl`
  - `data/zuco2/pickle/zuco2_source_4subj.pkl`
- 原始兼容目录：
  - `data/zuco2/`
  - `data/zuco2.0/`

从当前 `zuco2_source.pkl` 的结构看，样本按被试组织；当前默认样例文件中只包含 `YDG` 一个被试，共 390 个句子。每个句子样本包含：

- `content`：句子文本
- `sentence_level_EEG`：句级 EEG 统计特征
- `word`：逐词 EEG 特征列表
- `word_tokens_all` / `word_tokens_has_fixation` / `word_tokens_with_mask`：词级对齐辅助信息

每个词条目里包含：

- `content`：词文本
- `nFixations`：注视次数
- `word_level_EEG`：三类词级 EEG 特征
  - `GD`
  - `FFD`
  - `TRT`

### 2. 预处理方式

从 `EEG-To-Text/data.py` 和 `EEG-To-Text/util/construct_dataset_mat_to_pickle_v2.py` 看，当前预处理流程大致如下：

1. 从 `.mat` 文件读取句子内容、句级 EEG、词级 EEG。
2. 针对每个词，拼接 8 个频带特征：
   - `_t1`, `_t2`, `_a1`, `_a2`, `_b1`, `_b2`, `_g1`, `_g2`
3. 每个频带包含 105 维通道特征，因此单词输入维度为 `105 x 8 = 840`。
4. 对每个词向量做 1 维标准化。
5. 将一句话截断或补零到固定长度 `max_len=56`。
6. 生成：
   - `input_embeddings`：`[56, 840]` 的词级 EEG 序列
   - `input_attn_mask` / `input_attn_mask_invert`：有效词位置掩码
   - `target_ids`：目标文本经 tokenizer 编码后的 token 序列

需要说明的是，句级 EEG 特征在数据结构中被保留，但当前解码主链路实际使用的是词级 EEG 序列输入。

### 3. 标签 / 语义目标

当前仓库里有两类“目标”：

1. 解码目标：
   - 目标标签是句子原文 `content`
   - 本质上是一个条件文本生成任务
2. probing 目标：
   - 由 `probing/probe_tasks.py` 从词文本自动构造
   - 当前支持：
     - `pos`
     - `sentiment`
     - `frequency`
     - `length`

因此，项目里的“语义解析”并不只指最终生成文本，也包括对模型内部表征是否携带语言属性的分析。

## 模型与训练流程

### 1. 深度学习框架

当前实现基于：

- PyTorch
- Hugging Face Transformers

可见的预训练语言模型后端包括：

- `facebook/bart-large`
- `t5-large`
- `google/pegasus-xsum`

### 2. 模型结构 / 主要模块

核心代码位于：

- `EEG-To-Text/model_decoding.py`

当前主要有三类解码器实现：

1. `BrainTranslatorNaive`
   - 将 EEG 输入做线性映射：`840 -> 1024`
   - 直接对接 BART 的 `inputs_embeds`
   - 是当前本地轻量跑通脚本默认使用的版本

2. `BrainTranslator`
   - 在线性投影前增加 6 层 `TransformerEncoder`
   - 用于先建模 EEG 序列内部结构，再映射到语言模型嵌入空间

3. `T5Translator`
   - 结构类似，但后端替换为 T5
   - 会显式拼接任务前缀 embedding

### 3. 输入输出

- 输入：
  - 逐词 EEG 序列，形状近似为 `batch x 56 x 840`
- 输出：
  - 训练时输出 seq2seq loss
  - 推理时输出生成文本
  - probing 场景下可额外导出冻结表征，例如：
    - `encoder_last`
    - `projection_output`
    - `llm_input_embeddings`

### 4. 训练 / 推理流程

当前训练入口：

- `EEG-To-Text/train_decoding.py`

当前评测入口：

- `EEG-To-Text/eval_decoding.py`

训练流程要点：

1. 按 `unique_sent` 方式做划分，单被试内默认按 80% / 10% / 10% 划分 train/dev/test。
2. 用 tokenizer 将目标句子编码成文本标签。
3. 训练时将 EEG 序列映射到语言模型 embedding 空间，按生成任务优化。
4. 代码支持两阶段训练：
   - 第一阶段冻结大部分预训练参数
   - 第二阶段全量微调
5. 当前本地 smoke pipeline 默认跳过第一阶段，直接做少量 epoch 的轻量训练。

推理流程要点：

1. 加载训练配置和 checkpoint。
2. 用 `model.generate(...)` 生成文本，而不是只看 teacher forcing 下的 logits。
3. 输出目标文本、生成文本，并计算 BLEU / ROUGE / WER / CER。

## 推理 Demo

### 1. 当前 demo 能力

当前 demo 不是网页交互界面，而是命令行形式的离线推理 demo。它可以：

1. 读取已经整理好的 EEG pickle 数据。
2. 对测试集样本生成句子文本。
3. 把结果写入文本文件和评测文件。
4. 进一步导出中间表征，供 probing 脚本做语义属性分析。

### 2. 可以输入什么

- 已经整理成当前 schema 的 EEG 数据 pickle
- 典型路径：
  - `data/zuco2/pickle/zuco2_source.pkl`

当前不能直接输入：

- 原始脑电设备实时流
- 未对齐的原始 `.mat` 目录并立即在线解码
- 任意格式的自定义 EEG 文件

### 3. 可以输出什么

- 文本解码结果：
  - `EEG-To-Text/results/*.txt`
- 评测结果：
  - `EEG-To-Text/score_results/*.txt`
- 模型配置：
  - `EEG-To-Text/config/decoding/*.json`
- 训练 checkpoint：
  - `EEG-To-Text/checkpoints/...`
- probing 特征与分析结果：
  - `probing/outputs/...`

### 4. 如何运行

从项目根目录进入 `EEG-To-Text` 后，可直接运行本地轻量链路：

```bash
cd EEG-To-Text
./scripts/run_local_zuco2_pipeline.sh
```

这个脚本会依次执行：

1. `./scripts/train_decoding_local_zuco2.sh`
2. `./scripts/eval_decoding_local_zuco2.sh`

如果只想单独推理评测，可运行：

```bash
cd EEG-To-Text
./scripts/eval_decoding_local_zuco2.sh
```

### 5. 当前限制

1. 当前 demo 依赖离线整理后的数据格式，不支持实时采集和在线推理。
2. 本地默认脚本偏向 smoke test，样本数和 epoch 都被压缩，不代表完整训练设置。
3. 当前默认数据样例主要是单被试版本，不代表跨被试泛化能力。
4. 仓库里有 probing 结果和若干评测输出，但当前未形成一套统一、稳定、可直接对外宣称的系统量化结论。
5. 部分历史脚本与 checkpoint/config 命名存在演化痕迹，运行前应核对路径是否与当前文件一致。
6. 从现有 `probing` 结论看，跨被试泛化仍偏弱，因此更适合把它表述为“研究验证型 demo”，而不是成熟脑机产品。

## 我负责的工作

下面这部分按“当前仓库中可核实、适合写进个人项目经历的工程工作”来写；如果要用于正式简历，建议再按本人真实分工微调。

我主要负责了以下几类工作：

1. 数据处理与适配
   - 整理 ZuCo2 兼容目录
   - 维护 `.mat -> pickle` 的数据转换链路
   - 明确词级 EEG、句级 EEG 和文本标签之间的对应关系

2. 模型训练与推理链路打通
   - 基于 PyTorch + Transformers 搭建 EEG-to-Text 训练与评测流程
   - 使用 `BrainTranslatorNaive` / `BrainTranslator` 等结构完成解码实验
   - 修正并保留 `generate` 方式的推理评测链路

3. Demo 封装与运行脚本
   - 编写/整理本地运行脚本、runbook 和 checkpoint/config 组织方式
   - 支持按数据路径、checkpoint、配置文件做命令行调用

4. 结果导出与分析
   - 输出文本解码结果和 BLEU/ROUGE/WER/CER 评测文件
   - 冻结模型表征并接入 probing 分析脚本
   - 产出 JSON / CSV / Markdown 形式的分析结果，便于复盘和写作

5. 文档与工程整理
   - 对目录结构、关键脚本、运行方式和能力边界做文档化整理
   - 让项目既能作为研究型 demo 复现入口，也能作为简历项目材料使用

## 当前效果与限制

从当前仓库状态看，这个项目已经具备“数据可读入、模型可训练、结果可导出、表征可分析”的基本闭环，但仍应克制表述：

1. 该项目已经完成 demo 级别验证，不等于完成产品化或临床级验证。
2. 当前未做系统量化评测，不宜在简历或介绍中写精确准确率、数据规模或正式结论性指标。
3. 对外更稳妥的说法是：
   - 已完成基于公开 EEG 阅读数据的语义解码与表征分析 demo
   - 已打通训练、推理和分析链路
   - 跨被试泛化和真实场景可用性仍需继续验证

## 关键目录与运行方式

### 1. 关键目录

- 数据：
  - `data/zuco2/pickle/zuco2_source.pkl`
  - `data/zuco2/pickle/zuco2_source_4subj.pkl`
- 训练与推理主目录：
  - `EEG-To-Text/`
- probing 分析：
  - `probing/`
- 文档：
  - `docs/`

### 2. 关键脚本

- 数据转换：
  - `EEG-To-Text/util/construct_dataset_mat_to_pickle_v2.py`
  - `EEG-To-Text/scripts/prepare_dataset.sh`
- 数据读取：
  - `EEG-To-Text/data.py`
- 模型定义：
  - `EEG-To-Text/model_decoding.py`
- 训练入口：
  - `EEG-To-Text/train_decoding.py`
- 推理评测：
  - `EEG-To-Text/eval_decoding.py`
- 本地轻量运行：
  - `EEG-To-Text/scripts/run_local_zuco2_pipeline.sh`
  - `EEG-To-Text/scripts/train_decoding_local_zuco2.sh`
  - `EEG-To-Text/scripts/eval_decoding_local_zuco2.sh`
- probing：
  - `probing/extract_features.py`
  - `probing/probe_tasks.py`
  - `probing/train_probe.py`

### 3. 关键模型与输出文件

- 示例 checkpoint：
  - `EEG-To-Text/checkpoints/decoding_local/best/taskNRv2_finetune_BrainTranslatorNaive_skipstep1_b2_0_2_5e-05_5e-07_unique_sent_EEG.pt`
- 示例配置：
  - `EEG-To-Text/config/decoding/taskNRv2_finetune_BrainTranslatorNaive_skipstep1_b2_0_2_5e-05_5e-07_unique_sent_EEG.json`
- 示例结果：
  - `EEG-To-Text/results/taskNRv2-BrainTranslatorNaive-all_decoding_results.txt`
  - `EEG-To-Text/score_results/taskNRv2-BrainTranslatorNaive.txt`
- probing 输出：
  - `probing/outputs/exp_conclusion_package_v2_20260224/conclusion_package_v2.md`
  - `probing/outputs/exp_cross_subject_loso_4subj_20260224/cross_subject_loso_summary.json`

## 简历描述

### 1. 一句话版本

基于 PyTorch 和 Transformer 搭建脑电语义解析 demo，完成 EEG 阅读数据的预处理、文本解码、推理评测与表征 probing 分析链路。

### 2. 项目经历 bullet

- 基于公开 EEG 阅读数据构建脑电语义解析实验链路，将词级多频带 EEG 特征组织为可训练的序列输入。
- 使用 PyTorch 和 Hugging Face Transformers 实现 EEG-to-Text 解码模型，支持 BART/T5 等后端以及训练、评测、checkpoint 管理。
- 完成离线推理 demo，能够从整理后的 EEG pickle 数据生成文本结果，并输出 BLEU、ROUGE、WER、CER 等评测文件。
- 搭建冻结表征的 probing 流程，分析模型中间表示对词频、词长、词性及内容词/功能词等属性的可分性。
- 整理脚本、目录和运行文档，使项目可同时作为研究复现入口、RAG 知识库语料和简历项目材料使用。

## 面试 STAR 表达

### S：背景

我在做一个基于脑电信号的语义解析 demo，目标不是夸大“读心”，而是验证公开 EEG 阅读数据里是否存在可被深度学习模型利用的语言线索，并把训练、推理和分析流程工程化。

### T：任务

我的任务是把零散的数据、模型代码和实验脚本整理成一条可复现链路：既能把 EEG 数据转换成模型输入，完成句子解码，也能进一步分析模型内部表征是否保留了语义相关信息。

### A：行动

我先梳理了 ZuCo2 数据结构，明确句子、词、EEG 频带特征和文本标签的对应关系；然后基于 PyTorch 和 Transformers 搭建了解码模型训练与评测流程，补齐了本地运行脚本和 checkpoint/config 管理；在推理侧使用生成式解码输出文本结果；最后又增加了冻结表征后的 probing 分析，用统一脚本导出 JSON/CSV/Markdown 结果，便于复盘和写文档。

### R：结果

最终项目形成了一个研究型 demo：能够读取整理后的 EEG 数据，完成文本解码、结果导出和表征分析，说明这条技术路线在工程上是可跑通的。与此同时，我也明确保留了边界条件：当前未做系统量化评测，跨被试泛化仍有限，因此更适合将其表述为“脑电语义解析验证性 demo”，而不是成熟产品。
