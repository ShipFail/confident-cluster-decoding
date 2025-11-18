# Confident-Cluster Decoding (CCD)

Welcome to the official open‑source repository for **Confident‑Cluster Decoding (CCD)** — a new decoding paradigm designed for **reliable, deterministic, uncertainty‑aware, and hallucination‑resistant LLM agents**, with a special focus on **code generation**, **tool‑calling**, and **structured prediction**.

This repository hosts:

* the **research paper** (Quarto/LaTeX build system),
* the **reference implementation** of CCD,
* experiments and evaluation scripts,
* examples and integration patterns for coding agents.

---

# 🌟 Overview

Large Language Models (LLMs) often have multiple **high‑confidence alternatives** for the next token. Traditional greedy decoding forces commitment to exactly *one* of these choices — even when two or more tokens have nearly identical probability. Meanwhile, probabilistic decoding strategies (top‑k, top‑p, min‑p, typical decoding, etc.) introduce randomness that can lead to **hallucinations**, **schema violations**, or **invalid code**.

**Confident‑Cluster Decoding (CCD)** fills this gap.

CCD identifies a *cluster* of tokens whose probabilities are within a **relative threshold** of the top token. This “confident cluster” represents the model’s true uncertainty landscape and provides a basis for stable, reliable decoding.

CCD supports three modes:

* **CCD‑Deterministic** – Greedy output + cluster as an uncertainty signal.
* **CCD‑Cluster Sampling** – Sample only from high‑confidence tokens.
* **CCD‑Branching** – Explore multiple high‑confidence paths for validation, self‑consistency, or MBR selection.

CCD is **model‑agnostic**, **API‑friendly**, and naturally synergistic with coding agents, tool-call frameworks, and structured generation tasks.

---

# 🔥 Motivation & Story

### Why CCD Exists

While building *coding agents* and *REST‑schema tool‑calling systems*, we observed an important pattern:

* With **deterministic decoding** (`temperature=0`, `topK=1`), results were stable.
* But **rare instabilities** occurred when two tokens had almost identical probabilities.
* These instabilities were caused not by randomness, but by **floating‑point drift** and **logit tie‑breaking noise**.

This sparked the insight:

> *The model isn’t wrong — we are forcing it to pick a single winner when none exists.*

This led to the creation of **Confident‑Cluster Decoding**, a method that:

* Respects the model’s confidence structure.
* Avoids sampling from the low‑probability hallucination tail.
* Provides explicit **token‑level uncertainty signals**.
* Improves reliability for coding agents and structured tasks.

CCD combines the deterministic stability desired by production systems with the uncertainty‑awareness needed by intelligent agents.

---

# 📘 What’s in This Repository?

### 1. 📄 **The CCD Research Paper**

Located in `./paper/` and written in **Quarto Markdown**.

Includes:

* full academic write‑up,
* motivation & related work,
* CCD formal definitions,
* variants & algorithms,
* evaluation plan,
* bibliography.

You can build the PDF via:

```bash
the repo
quarto render ccd-paper.qmd
```

### 2. 🧠 **Reference Implementation**

Located in `./ccd/`.

Includes:

* CCD‑Deterministic implementation,
* CCD‑Cluster‑Sampling variant,
* CCD‑Branching (tree search) utilities,
* logprob‑based decoding utilities,
* adapters for Gemini / OpenAI / vLLM.

### 3. 🧪 **Experiments & Benchmarks**

Located in `./experiments/`.

Includes evaluation scaffolding for:

* code generation (HumanEval / MBPP),
* factual QA,
* GSM8K reasoning,
* JSON / structured generation validity tests.

### 4. 🛠️ **Coding Agent Integrations**

Located in `./examples/`.

Examples show how to use CCD in:

* code‑generation loops,
* tool‑calling pipelines,
* agent architectures that validate or branch on uncertainty signals.

---

# 📐 CCD — Core Concept

At decoding step *t*, the model defines a probability distribution over the vocabulary:

```
p_t(v) = P(token=v | context)
```

We identify the **top probability**:

```
P_max = max_v p_t(v)
```

Then define the **confident cluster**:

```
C_t(α) = { v | p_t(v) ≥ α · P_max }
```

Where 0 < **α** ≤ 1 controls how “tight” the cluster is.

### CCD Modes

* **Deterministic:** pick top token, record cluster.
* **Cluster Sampling:** sample from C_t(α).
* **Branching:** branch for each v ∈ C_t(α).

CCD offers **high determinism**, **high confidence**, and **explicit uncertainty**.

---

# 🚀 Why CCD Matters

### ✔ Reliable Code Generation

CCD drastically reduces:

* invalid syntax,
* broken JSON,
* wrong schema fields,
* tool call hallucination.

### ✔ Model‑Aware Uncertainty Estimation

Cluster size and distribution correlate strongly with error probability.

### ✔ Natural Fit for Agents

Agents can branch, validate, or abstain based on cluster structure.

### ✔ Production‑Friendly

CCD preserves deterministic behavior and avoids unstable probability tails.

---

# 📦 Installation & Usage

```bash
pip install confident-cluster-decoding
```

Basic usage:

```python
from ccd import ccd_decode
result = ccd_decode(model, prompt, alpha=0.85, top_k=15, mode="deterministic")
```

---

# 🤝 Contributing

We welcome:

* pull requests for algorithm improvements,
* implementation of new CCD variants,
* benchmark contributions,
* discussion and research collaboration.

Please open an issue or PR to join the effort.

---

# 🧭 Roadmap

* [ ] CCD‑Deterministic implementation
* [ ] CCD‑Cluster Sampling
* [ ] CCD‑Branching search engine
* [ ] Gemini & OpenAI adapters
* [ ] Full code-generation benchmark suite
* [ ] Visualization tools for clusters
* [ ] Paper submission to arXiv

---

# Author

Huan Li <https://github.com/huan>

# 📄 License

MIT License.

---

# 🌍 Community & Vision

CCD aims to become a **standard decoding method** for systems that require both **reliability** and **intelligence**.

This repository wants to:

* share the research,
* build the tools,
* and grow a community around CCD.

You’re invited to join.

Let’s build the next generation of safe, robust, and intelligent LLM agents — **together**.
