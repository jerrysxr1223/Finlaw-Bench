# Finlaw-Bench
# FinLaw: A Benchmark for Evaluating Retrieval-Augmented Reasoning in Financial Regulation

FinLaw is a large-scale benchmark designed to systematically evaluate **retrieval-augmented reasoning (RAG)** of large language models (LLMs) in **real-world financial regulatory scenarios**.  
It is grounded in **Chinese financial regulatory administrative penalty cases** and authoritative **financial laws and regulations**, and provides a **fine-grained, task-oriented evaluation framework**.

This repository contains the **benchmark dataset construction logic**, **retrieval-augmented inference pipeline**, and **evaluation code** used in our KDD 2026 submission.

---

## 🔍 Overview

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
