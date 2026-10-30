# LakehouseFusionAI: AI-Driven Data Lakehouse Optimization for Hybrid Cloud–Edge Analytics

**A Functional Research Reference Implementation for Intelligent, Cost-Aware and Latency-Aware Data Lakehouse Optimization**

## Overview

LakehouseFusionAI is a research-oriented implementation designed to investigate intelligent data lakehouse optimization in hybrid cloud–edge environments. The framework combines predictive resource estimation, Deep Q-Network (DQN)-based decision-making, adaptive workload placement, and multi-factor execution modelling to explore cost-efficient and latency-aware query execution.

The implementation provides two complementary components:

1. **Functional SQL Lakehouse Backend:** Implements SQLite-based ACID transactions, data validation, deduplicated ingestion, edge buffering, idempotent commits, synchronization, and local/cloud SQL operations.
2. **Hybrid Cloud–Edge Experiment Engine:** Models workload placement, resource queues, network bandwidth, round-trip time (RTT), caching, replication, service-level agreement (SLA) decisions, execution costs, and workload traces.

The framework provides a reproducible foundation for algorithm development, functional testing, and methodology validation.

## Key Features

- Hybrid cloud–edge workload execution modelling.
- Deep Q-Network (DQN)-based execution policy learning.
- Predictive resource estimation using multilayer perceptrons.
- Adaptive workload placement and feasible-action selection.
- Network bandwidth and RTT-aware transfer modelling.
- Stateful cache and replica management.
- SLA-aware execution decisions.
- Cost and latency estimation.
- SQLite-based ACID-compliant data storage.
- Edge data ingestion, buffering, and synchronization.
- Dataset preparation adapters for public datasets.
- Matched baseline comparisons and ablation experiments.
- Paired Wilcoxon statistical testing with Holm correction.
- Automated functional testing.
- High-resolution scientific figure generation.

## Framework Architecture

The implementation consists of two distinct execution paths.

### 1. Functional SQL Data Path

The storage component provides actual SQL-based data operations through SQLite.

Its capabilities include:

- Transaction-based data ingestion.
- ACID transaction support.
- Input data validation.
- Duplicate detection and deduplicated ingestion.
- Edge-side buffering.
- Idempotent synchronization commits.
- Local and cloud SQL reads.
- Query correctness validation.

### 2. Logical Hybrid Execution Engine

The experiment engine provides a stateful simulation of hybrid cloud–edge execution.

It models:

- Edge and cloud resource availability.
- Workload placement decisions.
- Resource queue contention.
- Network transfer delays.
- Bandwidth and RTT effects.
- Cache and replica state transitions.
- SLA satisfaction decisions.
- Execution cost and latency.
- Workload execution traces.

**Scientific implementation note:** The hybrid execution engine is a logical emulation. It does not constitute a physical Spark, Delta Lake, or MinIO deployment. Its latency, throughput, and cost values are model-generated outcomes rather than measurements from an actual distributed cloud–edge infrastructure.

## Algorithmic Components

### Algorithm 1: Edge Data Ingestion and Synchronization

Implemented primarily through `storage.EdgeAgent` and the logical resource modelling provided by `core.HybridEnvironment`.

The storage component handles real ingestion, buffering, and synchronization operations, while the experiment engine models resource and network-transfer effects.

### Algorithm 2: Predictive Resource Estimation and DQN-Based Decision-Making

The learning component incorporates:

**Predictive Models**
- Two multilayer perceptron (MLP) regressors.
- Hidden-layer architecture: 128 and 64 neurons.

**Deep Q-Network**
- Hidden-layer architecture: 128 and 64 neurons.
- Epsilon-based exploration.
- Target network updates every 200 gradient updates.
- Replay memory capacity: 50,000 transitions.
- Discount factor: 0.95.
- Batch size: 64.
- Adam optimizer with a learning rate of 0.0001.
- Gradient clipping.

The evaluation component performs feasible-action policy selection.

### Algorithm 3: Hybrid Execution and Resource Optimization

Implemented through `core.HybridEnvironment.step`.

The logical execution engine models:

- Workload placement.
- Local and cloud computation.
- Caching and replication.
- Network transfer costs.
- Resource contention.
- SLA-related execution decisions.

Actual SQL operations are independently executed through `storage.SQLLakehouse`.

## Supported Datasets

The framework provides data preparation adapters for three public datasets.

| Dataset | Data Category | Application |
|---|---|---|
| NYC Taxi Trip Records | Batch data | Batch workload preparation |
| DEBS 2014 Smart Home | Streaming data | Smart-home event preparation |
| GDELT | Event data | Event-oriented data preparation |

### Official Dataset Sources

- NYC Taxi and Limousine Commission: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- GDELT Project: https://www.gdeltproject.org/data.html
- DEBS Grand Challenges: https://debs.org/grand-challenges/

**Note:** Dataset files must be obtained separately from their official sources. The repository does not include the complete original datasets. The `prepare` command performs actual dataset transformations, whereas the current evaluation command generates deterministic logical workload descriptors rather than directly executing experiments on the prepared dataset records.

## Project Structure

```text
LakehouseFusionAI/
│
├── main.py
├── requirements.txt
│
├── lakehousefusion/
│   ├── __init__.py
│   ├── core.py
│   ├── learning.py
│   ├── evaluation.py
│   ├── storage.py
│   ├── datasets.py
│   └── plots.py
│
├── tests/
│
├── data/
│
├── results/
│
└── checkpoints/
```

### Core Modules

| Module | Description |
|---|---|
| `core.py` | Hybrid environment, workloads, states, action masks, resource queues, and reward modelling. |
| `learning.py` | Predictive MLP regressors and DQN implementation. |
| `evaluation.py` | Baseline comparisons, ablation experiments, summaries, and statistical analysis. |
| `storage.py` | SQLite-based ACID storage, edge ingestion, buffering, and synchronization. |
| `datasets.py` | CSV and Parquet dataset preparation adapters. |
| `main.py` | Command-line interface for training, evaluation, preparation, and demonstrations. |
| `plots.py` | Experimental figure generation. |
| `tests/` | Automated functional tests. |

## Installation and Requirements

**Recommended Python Version:** Python 3.11

### Step 1: Clone the Repository

```bash
git clone https://github.com/datasph/LakehouseFusionAI 
cd LakehouseFusionAI
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Execution Guide

### 1. SQL Lakehouse Demonstration

```bash
python main.py sql-demo
```

Executes the functional SQL data-path demonstration.

### 2. Model Training

```bash
python main.py train --n 350 --epochs 3
```

A shorter training run can be executed using:

```bash
python main.py train --n 100 --epochs 1
```

Training and evaluation use isolated random seeds. The training seed is 42, while evaluation seeds start from 10001.

### 3. Normal Network Evaluation

```bash
python main.py evaluate --n 200 --runs 10 --qps 50 --network normal
```

### 4. Constrained Network Evaluation

```bash
python main.py evaluate --n 200 --runs 10 --qps 200 --network constrained --out results/stress --ablation
```

The evaluation process generates:

- `run_metrics.csv`
- `query_traces.csv`
- `summary.csv`
- `statistics.csv`
- `metadata.json`

### 5. Dataset Preparation

**NYC Taxi:**

```bash
python main.py prepare --dataset taxi --input data/raw/taxi.csv --out data/processed/taxi.csv
```

**DEBS 2014:**

```bash
python main.py prepare --dataset debs --input data/raw/debs.csv --out data/processed/debs.csv
```

**GDELT:**

```bash
python main.py prepare --dataset gdelt --input data/raw/gdelt.csv --out data/processed/gdelt.csv
```

For Parquet input/output support, install:

```bash
pip install pyarrow
```

### 6. Generate Experimental Figures

```bash
python main.py plot --metrics results/run_metrics.csv --out results/figures
```

The plotting command generates five 600-DPI PNG charts with explicitly labelled modelled outcomes.

### 7. Run Automated Tests

```bash
python -m pytest tests -q
```

## Experimental Evaluation

The evaluation component supports comparisons between different execution strategies using matched incoming query traces for each seed.

The evaluation framework includes:

- Run-level performance metrics.
- Query-level execution traces.
- Latency analysis.
- Cost analysis.
- Throughput analysis.
- SLA satisfaction analysis.
- Baseline comparisons.
- Ablation experiments.
- Paired two-sided Wilcoxon statistical testing.
- Holm multiple-comparison correction within each condition and metric.

**Important:** Statistical significance obtained from the logical experiment engine does not establish production-level performance improvements. Latency, throughput, and cost are modelled rather than physically measured against external infrastructure.

The simulator is sequential and event-driven. Offered-load QPS and modelled-completion throughput QPS represent different quantities.

## Implementation Scope and Limitations

For scientific transparency, the following implementation boundaries must be considered:

1. The repository does not implement a physical Delta Lake, Apache Spark, MinIO, Docker orchestration, or Linux traffic-control environment.
2. The specified four-edge and one-cloud physical testbed is not included.
3. Real remote multi-host partition transfers and physically measured network bandwidth are not implemented.
4. Cache behaviour is represented through stateful LRU metadata and modelled savings.
5. The SQLite backend does not execute cache retrieval through the logical experiment model.
6. Predictor training uses explored actions rather than all counterfactual actions.
7. The `no_edge_preprocess` ablation is a simplified diagnostic.
8. Synthetic time and cost coefficients require calibration using actual measurements.
9. The dataset preparation adapters and logical evaluation engine are not currently integrated into a complete dataset-driven experimental pipeline.
10. The supplied implementation is not a verified reproduction of the manuscript's numerical results.

These limitations distinguish the functional research reference implementation from a complete physical cloud–edge deployment.

## Reproducibility

The framework supports reproducible training and evaluation through:

- Controlled random seeds.
- Isolated training and evaluation environments.
- Consistent query traces across method comparisons.
- Saved model checkpoints.
- Structured CSV experiment outputs.
- JSON experiment metadata.
- Automated functional tests.
- High-resolution experimental figures.

Reproducing the manuscript's physical benchmark results requires implementing and validating the specified infrastructure and calibrating the model against actual measurements.

## Research Context

**Research Title:**

*AI-Driven Data Lakehouse Optimization Reduces Cost and Latency in Hybrid Cloud–Edge Analytics*

This repository provides a functional research reference implementation aligned with the research methodology. It supports algorithm development, logical experiment evaluation, SQL storage validation, and reproducibility-oriented experimentation.

## Citation

If you use this implementation in academic research, please cite the associated research work after its publication.

```text
Title: AI-Driven Data Lakehouse Optimization Reduces Cost and Latency in Hybrid Cloud–Edge Analytics
Year: 2026
DOI: [Add DOI when available]
```

## Acknowledgements

The implementation uses publicly available dataset sources and established machine-learning and data-processing libraries. Dataset authors and providers retain their respective rights and licensing terms.

## License

This project is licensed under the MIT License.

---

