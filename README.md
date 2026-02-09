
# FinLaw: A Benchmark for Evaluating Retrieval-Augmented Reasoning in Financial Regulation

FinLaw is a large-scale benchmark designed to systematically evaluate **retrieval-augmented reasoning (RAG)** of large language models (LLMs) in **real-world financial regulatory scenarios**.  
It is grounded in **Chinese financial regulatory administrative penalty cases** and authoritative **financial laws and regulations**, and provides a **fine-grained, task-oriented evaluation framework**.

This repository contains the **benchmark dataset construction logic**, **retrieval-augmented inference pipeline**, and **evaluation code** used in our KDD 2026 submission.

---

## 🔍 Overview

<img width="3953" height="2360" alt="引言图" src="https://github.com/user-attachments/assets/fb38b1a2-e03f-4d9b-96ad-bbcac9116e8a" />
<img width="3953" height="2360" alt="引言图" src="https://github.com/user-attachments/assets/fb38b1a2-e03f-4d9b-96ad-bbcac9116e8a" />

Financial regulation is a highly rule-intensive and high-risk domain. While Retrieval-Augmented Generation (RAG) is widely used to enhance LLMs with external knowledge, **its actual effectiveness varies substantially across regulatory reasoning tasks**.

FinLaw aims to answer three core research questions:

- **RQ1**: What are the capability boundaries of LLMs on financial regulatory tasks *without* external legal knowledge?
- **RQ2**: How does retrieval augmentation affect different types of regulatory reasoning tasks?
- **RQ3**: Why does retrieval augmentation help some tasks but fail on others?

To this end, FinLaw decomposes regulatory decision-making into **four representative tasks** and evaluates models under both **retrieval-free** and **retrieval-augmented** settings.

---

## 📊 Tasks

Each sample requires the model to jointly predict the following four elements:

1. **Penalty Type**  
   Multi-label classification over 27 administrative penalty categories.

2. **Violation Domain**  
   Multi-label classification over 17 financial violation domains.

3. **Legal Basis**  
   Article-level identification of applicable laws and clauses  
   (e.g., `《中华人民共和国银行业监督管理法》第四十六条`).

4. **Penalty Level**  
   Severity assessment into five levels: **A / B / C / D / E**,  
   derived from fine and confiscation amount intervals.

---

## 🗂 Dataset

### Data Sources

FinLaw consists of two main components:

- **Violation & Penalty Dataset**
  - 36,408 administrative penalty cases (2016–2025)
  - Publicly released by the National Financial Regulatory Administration of China and its provincial branches
  - Focus on the banking sector

- **Laws & Regulations Dataset**
  - Authoritative financial regulatory texts
  - Laws, administrative regulations, departmental rules, and normative documents
  - Collected from official regulatory websites and legal information platforms

All data are publicly available and processed following a strict pipeline to ensure **authority, traceability, and reproducibility**.

---

### Data Processing & Annotation

#### Penalty Data
From raw disclosure tables, we extract and enrich:

- Involved party & party type  
- Violation facts  
- Violation domain  
- Penalty content  
- Legal basis  
- Penalty type  
- Fine & confiscation amount  
- Affiliated head office  
- Penalty level (A–E)

#### Legal Texts
- Clause-level segmentation using rule-based parsing
- Each clause stored with full citation path:
  - *Law name – chapter – article number*
- Encoded into **1024-dim embeddings** using `text-embedding-v3`
- Stored in **ChromaDB** for semantic retrieval

---

## 🔄 Inference Pipeline

1. **Input Construction**
   - Violation facts
   - Party information (anonymized)
   - Regulatory authority
   - Dates

2. **Legal Retrieval**
   - Retrieve **Top-5** most relevant legal clauses from the vector database
   - Check whether ground-truth legal basis is included

3. **Gold Injection Strategy**
   - If retrieval misses the true legal basis:
     - Automatically inject the gold legal clause into the candidate set
   - Ensures fair comparison between retrieval success and reasoning capability

4. **LLM Inference**
   - Models must output a **strict JSON object**:
     ```json
     {
       "处罚类型": "...",
       "违规领域": "...",
       "法律依据": "...",
       "处罚等级": "A/B/C/D/E"
     }
     ```

---

## 📐 Evaluation Protocol

FinLaw adopts a **hierarchical and fine-grained evaluation framework**.

### Sample-Level Scoring

| Task | Scoring |
|----|----|
| Penalty Type | 2 = exact match / 1 = partial / 0 = wrong |
| Violation Domain | 2 / 1 / 0 |
| Legal Basis | 2 = law+article correct / 1 = law correct / 0 = wrong |
| Penalty Level | 1 = correct / 0 = wrong |

---

### Aggregated Metrics

- **Penalty Type & Violation Domain**
  - Exact / Partial / Error rates
  - Precision, Recall, F1
  - Jaccard similarity
  - Hamming Loss

- **Legal Basis**
  - Exact / Partial / Error rates
  - Law name hit rate
  - Law–article pair Precision / Recall / F1

- **Penalty Level**
  - Accuracy
  - Macro-F1 / Micro-F1
  - Per-level Precision / Recall / F1

This design explicitly separates **retrieval errors** from **generation-level reasoning failures**.

---

## 🧪 Evaluated Models

We evaluate both open-source and proprietary LLMs:

- GPT-5.2  
- Gemini 3 Flash  
- DeepSeek-V3.2  
- Kimi-k2.5 (reasoning model)  
- Qwen3-max (reasoning model)

All models are tested under:
- **Without Retrieval** (parametric knowledge only)
- **With Retrieval** (vector database + gold injection)

---
**Table: Full evaluation results for all models.**  
Each model reports the main experiment (with vector database) and the control experiment (without vector database).

---

**Table: Full evaluation results for all models.**  
Each model reports the main experiment (with vector database) and the control experiment (without vector database).

---

### 🟦 Penalty Type

| Metric | Qwen3-max (Main) | Qwen3-max (Control) | Gemini3-Flash (Main) | Gemini3-Flash (Control) | GPT-5.2 (Main) | GPT-5.2 (Control) | Kimi-K2.5 (Main) | Kimi-K2.5 (Control) | DeepSeekV3.2 (Main) | DeepSeekV3.2 (Control) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact Match Accuracy | 0.009 | 0.007 | **0.422*** | 0.363 | 0.081 | 0.078 | 0.187 | 0.230 | 0.102 | 0.087 |
| Partial Match Accuracy | **0.927*** | 0.928 | 0.517 | 0.576 | 0.843 | 0.865 | 0.759 | 0.718 | 0.842 | 0.860 |
| Error Rate | 0.064 | 0.065 | 0.061 | 0.061 | 0.076 | 0.056 | 0.054 | **0.052*** | 0.055 | 0.053 |
| Precision | 0.374 | 0.372 | **0.672*** | 0.644 | 0.418 | 0.420 | 0.482 | 0.502 | 0.453 | 0.436 |
| Recall | 0.875 | 0.874 | 0.839 | 0.839 | 0.858 | **0.886*** | 0.875 | 0.868 | 0.872 | 0.881 |
| F1-score | 0.524 | 0.522 | **0.746*** | 0.729 | 0.562 | 0.569 | 0.622 | 0.636 | 0.596 | 0.583 |
| Average Jaccard Index | 0.368 | 0.365 | **0.680*** | 0.650 | 0.428 | 0.431 | 0.515 | 0.542 | 0.464 | 0.446 |
| Hamming Loss | 0.632 | 0.635 | **0.320*** | 0.350 | 0.572 | 0.569 | 0.485 | 0.458 | 0.536 | 0.554 |

---

### 🟦 Violation Domain

| Metric | Qwen3-max (Main) | Qwen3-max (Control) | Gemini3-Flash (Main) | Gemini3-Flash (Control) | GPT-5.2 (Main) | GPT-5.2 (Control) | Kimi-K2.5 (Main) | Kimi-K2.5 (Control) | DeepSeekV3.2 (Main) | DeepSeekV3.2 (Control) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact Match Accuracy | 0.260 | 0.150 | **0.675*** | 0.675 | 0.217 | 0.221 | 0.548 | 0.563 | 0.608 | 0.528 |
| Partial Match Accuracy | 0.702 | **0.824*** | 0.263 | 0.257 | 0.740 | 0.749 | 0.388 | 0.374 | 0.310 | 0.396 |
| Error Rate | 0.038 | **0.026*** | 0.069 | 0.068 | 0.043 | 0.029 | 0.064 | 0.063 | 0.082 | 0.076 |
| Precision | 0.521 | 0.469 | **0.809*** | 0.809 | 0.559 | 0.567 | 0.711 | 0.721 | 0.755 | 0.700 |
| Recall | 0.935 | **0.955*** | 0.852 | 0.853 | 0.931 | 0.943 | 0.879 | 0.878 | 0.845 | 0.860 |
| F1-score | 0.669 | 0.629 | **0.830*** | 0.830 | 0.699 | 0.708 | 0.786 | 0.792 | 0.798 | 0.772 |
| Average Jaccard Index | 0.573 | 0.504 | **0.804*** | 0.804 | 0.576 | 0.588 | 0.741 | 0.749 | 0.762 | 0.723 |
| Hamming Loss | 0.427 | 0.496 | **0.196*** | 0.196 | 0.424 | 0.412 | 0.259 | 0.251 | 0.238 | 0.277 |

---

### 🟦 Legal Basis

| Metric | Qwen3-max (Main) | Qwen3-max (Control) | Gemini3-Flash (Main) | Gemini3-Flash (Control) | GPT-5.2 (Main) | GPT-5.2 (Control) | Kimi-K2.5 (Main) | Kimi-K2.5 (Control) | DeepSeekV3.2 (Main) | DeepSeekV3.2 (Control) |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact Match Accuracy | **0.541*** | 0.073 | 0.478 | 0.266 | 0.523 | 0.052 | 0.513 | 0.219 | 0.442 | 0.129 |
| Partial Match Accuracy | 0.395 | **0.812*** | 0.457 | 0.650 | 0.320 | 0.870 | 0.426 | 0.689 | 0.384 | 0.759 |
| Error Rate | 0.064 | 0.115 | 0.065 | 0.084 | 0.157 | 0.078 | **0.061*** | 0.092 | 0.174 | 0.112 |
| Law Name Hit Rate | **0.936*** | 0.885 | 0.935 | 0.916 | 0.843 | 0.922 | 0.939 | 0.908 | 0.826 | 0.888 |
| Article Pair Precision | 0.764 | 0.394 | 0.748 | 0.584 | 0.741 | 0.321 | **0.786*** | 0.522 | 0.748 | 0.441 |
| Article Pair Recall | 0.669 | 0.555 | **0.682*** | 0.644 | 0.624 | 0.540 | 0.659 | 0.639 | 0.566 | 0.522 |
| Article Pair F1-score | 0.713 | 0.461 | 0.714 | 0.613 | 0.678 | 0.403 | **0.717*** | 0.575 | 0.644 | 0.478 |

---

### 🟦 Penalty Level

| Metric | Qwen3-max (Main) | Qwen3-max (Control) | Gemini3-Flash (Main) | Gemini3-Flash (Control) | GPT-5.2 (Main) | GPT-5.2 (Control) | Kimi-K2.5 (Main) | Kimi-K2.5 (Control) | DeepSeekV3.2 (Main) | DeepSeekV3.2 (Control) |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 0.149 | 0.151 | **0.170*** | 0.170 | 0.150 | 0.147 | 0.149 | 0.149 | 0.149 | 0.148 |
| Macro F1-score | 0.103 | 0.108 | **0.181*** | 0.181 | 0.093 | 0.095 | 0.112 | 0.108 | 0.078 | 0.071 |
| Micro F1-score | 0.214 | 0.216 | **0.245*** | 0.245 | 0.214 | 0.212 | 0.213 | 0.214 | 0.213 | 0.213 |

---

*`*` indicates the best result under the retrieval-augmented (Main) setting for each metric.*


---

## 📁 Repository Structure

```text
.
├── data/
│   └── data_finlaw.csv              # Structured penalty dataset
├── vector_db/
│   └── chroma_db_qwen/              # Clause-level legal vector database
├── experiments/
│   ├── Main_experiment.py           # Retrieval-augmented inference pipeline
│   └── evaluat.py                   # Evaluation metrics and scoring
├── outputs/
│   ├── experiment_results.jsonl
│   └── evaluation_metrics.csv
├── README.md
