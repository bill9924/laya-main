# laya-main
Using laya (reproducing Jev) to align patents with concepts.

---

## Overview

将专利文本与 OpenAlex Concepts 进行语义对齐。整体思路：用 Jev（TypeSafe 决策式模型）替代生成式 LLM，对候选概念做判别式选择，完成专利到概念的映射。

本仓库使用 **laya**（社区复现的 Jev）作为判别模型。

---

## Pipeline

```
Google Patent (BigQuery)
        │
        ▼
  patent.json                        # title + description
        │
        ▼
  TF-IDF + 余弦相似度                 # 候选集构造
        ▲
        │
  openalex_concepts.jsonl            # 本地缓存的 OpenAlex Concepts（约 6.5 万条）
        │
        ▼
  laya (Jev 复现) choice             # 判别式选择
        │
        ▼
  jev_concept_results.json           # 最终对齐结果
```

1. 从 Google Patent（https://console.cloud.google.com/bigquery）下载专利，构造 title + description 字段，保存至 `dataset/patent.json`
2. OpenAlex Concepts 无法直接实现 description 匹配，故将 OpenAlex 全部约 **6.5 万个** Concepts 下载到本地，保存至 `openalex_concepts_data/openalex_concepts.jsonl`
3. 使用 **TF-IDF + 余弦相似度排序**，构造候选集
4. 使用复现 Jev 的 **laya 模型**进行 choice
5. 最终结果保存在 `jev_concept_data/jev_concept_results.json`

---

## Directory Structure

```
laya-main/
├── dataset/
│   └── patent.json                     # Google Patent 下载的 title + description
├── openalex_concepts_data/
│   └── openalex_concepts.jsonl         # 本地缓存的 OpenAlex Concepts（约 6.5 万条）
└── jev_concept_data/
    └── jev_concept_results.json        # Jev 最终对齐结果
```

---

## Data

### Patents

- 来源：https://console.cloud.google.com/bigquery
- 选取 **G06N** 大类
- 网页端限制下载为 10MB，所以数据集规模为 **9183 条**消息

### Concepts

- OpenAlex Concepts 是已经存在的概念库，直接调用 OpenAlex API 进行检索
- 但 OpenAlex 的 Concepts（旧体系）已经于 **2024 年被正式弃用**
- 当前标准是 **Topics（新体系）**，topics 无法输出，是相对聚合的，代表一个研究领域，所以没有 concepts 准确
- 因此这里仍使用**没有维护的旧体系**

---

## Why TF-IDF instead of API query

因为使用 OpenAlex Concepts 的 API，出现了 description 构造出来的查询向量 query 无法检索出来的问题。核心在于一长串的表达，无法匹配到 concepts。

于是：

- 把 OpenAlex 全部约 **6.5 万个 Concepts** 原样下载到本地缓存——这仍然是真实的 OpenAlex Concepts 数据，不是自建分类体系，只是换了个获取方式。
- 因为是一句话，没有上下文逻辑，并且客观上，title 和 description 都是作者基于文章核心观点列出来的，简洁明了，且为了快捷计算，使用 **TF-IDF**。
- 改成对本地缓存做 **TF-IDF + 余弦相似度排序**，不再对每条专利单独打 concepts。这样每条 title + description 非空的专利都保证拿到 top_k 个候选。

---

## Output Format

```json
{
  "patent_id": "JP-2026052844-A",
  "title": "System, inference model generation method, and inference model generation program",
  "concept": "Image synthesis",
  "confidence": 0.0525
}
```

---

## Performance

- 先跑 100 条，时间为：**5s**，非常快
- 跑通全部 9183 条

---

## Example

**专利：**

```json
{
  "title": "System, inference model generation method, and inference model generation program",
  "description": "[Problem] To more easily generate an inference model that performs inference on a workpiece. [Solution] The system comprises an image generation unit that generates multiple workpiece images showing workpieces viewed from different viewpoints, an image synthesis unit that generates one or more virtual random stack images showing multiple randomly stacked workpieces based on the multiple workpiece images, and a learning unit that trains an inference model that infers workpiece information about one or more workpieces shown in the random stack image based on the one or more virtual random stack images. [Selected Figure] Figure 1"
}
```

**TF-IDF 候选结果（节选）：**

```json
"candidate_concepts": [
  {
    "id": "https://openalex.org/C162376815",
    "display_name": "Frequentist inference",
    "description": "statistical inference based on frequency and proportion in sample data",
    "level": 4,
    "works_count": 24825,
    "cited_by_count": 393672,
    "wikidata": "https://www.wikidata.org/wiki/Q2158281",
    "score": 0.284186
  },
  {
    "id": "https://openalex.org/C2779793024",
    "display_name": "Indirect Inference",
    "description": null,
    "level": 3,
    "works_count": 2160,
    "cited_by_count": 21005,
    "wikidata": "https://www.wikidata.org/wiki/Q17299941",
    "score": 0.27669
  },
  {
    "id": "https://openalex.org/C2989087649",
    "display_name": "Image synthesis",
    "description": "process of generating an image from a model",
    "level": 3,
    "works_count": 8553,
    "cited_by_count": 49341,
    "wikidata": "https://www.wikidata.org/wiki/Q176953",
    "score": 0.226465
  }
]
```

**Jev 最终选择：**

```json
{
  "patent_id": "JP-2026052844-A",
  "title": "System, inference model generation method, and inference model generation program",
  "concept": "Image synthesis",
  "confidence": 0.0525
}
```

Jev 认为判断更加准确，没有根据 TF-IDF 哪个分更好去选择，且速度很快。

---

## Background: Jev / TypeSafe

### 决策式 vs 生成式

TypeSafe 不再像 LLM 那样是生成式的，而是 **Jev 是"决策式"的**，直接输出结构化的答案 + 概率。

Benchmark 条件：Same 27 questions. Same order. 27 QUESTIONS • ONE REQUEST EACH • STARTED TOGETHER

### 回答任务类型

| 类型 | 示例 | 输出含义 |
| --- | --- | --- |
| 是否问题 | `Q: Revenue currently impacted?` | `{ "noul": 0.85, "type": "noul" }`，noul 为 0~1 的数，表示答案为"是"的概率 |
| 选择问题 | `Q: Which incident scope?` | `{ "choice": "single_account", "confidence": 0.75 }`，即分类问题 |
| 数值等级问题 | `Q: Churn likelihood level?` | `{ "score": 1.6, "confidence": 0.6 }`，客户流失可能性等级为 1.6 |

### 使用方式

- Jev 做决策的先决条件是：**你得先给它"可判断的范围"**
- Jev 擅长的是：在已有信息和候选里，帮你做判断；**不适合凭空替你想出所有可能答案**

### 官方宣传优点

- Jev 产生类型决策，更像代码：可靠、快速、自洽且类型安全
- 每一个 Jev 决策都附带信心估计，软件可以在信心高时采取行动，信心低时升级
- 不取悦人
- cost 少
- 擅长常识判断：内容分类、评分响应、评估信息，以及路由请求
- 新的 Workflow Evals：固定锁死一套流程，采用最聪明的两个模型答案当作参考答案

### 双系统定位

官方借用认知科学的"双系统"概念：

- **System 2**：昂贵的前沿大模型，负责慢速的深度推演
- **System 1**：Jev，充当快速廉价的反射弧

在需要做前置过滤或路由分流时，用小判别模型就能搞定，不需要让生成模型全程陪跑。

### 已知局限

- **缺少思维链**：单步前向推导无法完成多层因果逻辑推理。在钓鱼邮件基准测试 `anisselbd/jev-phishing-bench` 中，面对包含多层转折与伪造身份的诱骗邮件，支持思维链推理的 Claude Haiku 判定准确率明显优于 Jev。
- 当参考依据放在候选项之后时，正确率维持在较高水平。
- 这些局限的共同根源在于**单步前向计算的固定容量**。单个前向网络只能在固定深度的矩阵变换中处理特征。

---

## Reproducing Jev: Open-source Implementations

Jev 目前尚未完全开源，但结合官方技术报告与开源社区的逆向复现，实现路径大致有两条。开源社区在发布两小时内就跑通了两条复现路径，都不需要重写注意力机制，直接在开源小模型上就能运行。

### 路径一：把生成模型"截断"在生成之前

拿一个已经训练好的 LLM，把它原本就会产生的"下一 Token 概率分布"拿出来使用。

- **Logits 投影**——直接从 LLM 的"下一词预测"里拿分。模型跑完一次，最后会输出一堆"下一个词的概率分数"（logits）。只看候选答案对应的那几个词的分数，然后做 Softmax 归一化，得到置信度。

### 路径二：把模型设计成分类器

把上下文当作"前提"，候选动作当作"假设"，让模型判断"前提能不能推出假设"。

- 加一个分类头，大模型负责提取特征，分类头负责把特征变成类别
- 不再使用 LM Head（用来预测下一个 Token）进行 Token 生成，而是在 Transformer 的隐藏表示上接一个 Classification Head，通过训练直接把 transformer 算出来的向量，变成三维的：蕴含、矛盾、中立

### 已有复现的开源模型

| 序号 | 链接 | 路径 |
| --- | --- | --- |
| 1 | https://github.com/featherless-ai/simple-jev | 第一条 |
| 2 | https://github.com/Zefan-Cai/Open-Jev | 第二条变体：把上下文 + 问题 + 某个候选答案拼成一个序列，直接给 choice 的选项，noul 的是否，score 等级打分 |
| 3 | https://github.com/TianyuCodings/NanoJev | 第二条：针对迷宫游戏，对候选路径打分/编码 |
| 4 | https://github.com/GPT-AGI/OpenJev | 第一条：One forward pass. Logits are read at the answer position and softmaxed over your labels only. Nothing outside the option set can win. No decoding loop. |
| 5 | https://github.com/Yinsongxu/LLM2Jev | 第一条：在 prefill 阶段读取 logits 计算概率并直接组装结果，9月22日支持多模态 |
| 6 | https://huggingface.co/v6543210/openJev-1.5B | 第一条 |
| 7 | https://github.com/rupeshpoojary9/poorjev | — |
| 8 | https://github.com/Shalimov04/open-jev | 蒸馏模型，teacher 直接拿 LLM 输出的 logits，student 推理时完全不碰 teacher 的 logits，也不调用 teacher |
| 9 | https://github.com/AppitStudio/awesome-jev | — |
| 10 | https://github.com/AbdelStark/awesome-typesafe-jev | — |
| 11 | https://github.com/ekzhang/openjev-sglang | — |
| 12 | https://github.com/wfzyx/von | — |
| 13 | https://github.com/vinnylarouge/jevlike | — |
| 14 | https://github.com/jaredpalmer/kev | 3.6k⭐ |
| 15 | https://github.com/TianyuCodings/NanoJev | 2k⭐ |
| 16 | https://github.com/NandhaKishorM/laya | 最大竞争对手，9.22 日 15.8k⭐ |

### 官方 API

官方教程：https://docs.typesafe.ai/introduction/quickstart#call-it-the-api

> 备注：官方 API 页面曾出现"满了，登不进去"的情况。

---

## Notes on Classification Strategy

分类存在"每一个层级都分类，会给出最可能的一个选项"的问题，但会出现**硬选**的情况，最后一定会一直到最底层。

也就是说：OpenAlex Concepts 是一棵层级树，但**专利并不一定应该被分类到叶子节点**。

所以需要根据专利的现有描述，判断**最多支持到哪个 Concept**。

> 纠正：通过树节点剪枝操作，还是太慢了。所以最后的路线定位为上述 Pipeline。
