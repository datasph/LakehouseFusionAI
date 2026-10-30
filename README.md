[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23796788.svg)](https://doi.org/10.5281/zenodo.23796788)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/datasph/LakehouseFusionAI)


[DOI](https://doi.org/10.5281/zenodo.23796788) [GitHub](https://github.com/datasph/LakehouseFusionAI)

# LakehouseFusionAI: AI-Driven Data Lakehouse Optimization for Hybrid Cloud–Edge Analytics

**A functional research reference implementation for cost-aware and latency-aware data lakehouse optimization**

## Overview

LakehouseFusionAI is a research-oriented software implementation for investigating intelligent data lakehouse optimization in hybrid cloud–edge environments. It combines predictive resource estimation, Deep Q-Network (DQN)-based decision-making, adaptive workload placement, and multi-factor execution modelling.

The repository contains two complementary components:

1. **Functional SQL lakehouse backend:** SQLite-based ACID transactions, input validation, deduplicated ingestion, edge buffering, idempotent commits, synchronization, and SQL operations.

2. **Hybrid cloud–edge experiment engine:** Stateful logical modelling of workload placement, resource queues, bandwidth, round-trip time (RTT), caching, replication, service-level agreement (SLA) outcomes, costs, and workload traces.

The implementation supports algorithm development, functional testing, and controlled experimentation. The two components have different evidence and reproducibility scopes, described below.

## Relationship to the Research Manuscript

**Associated manuscript:** *AI-Driven Data Lakehouse Optimization Reduces Cost and Latency in Hybrid Cloud–Edge Analytics*.

The manuscript presents LakehouseFusionAI as an integrated hybrid cloud–edge data lakehouse architecture with learning-based prediction, DQN execution planning, data ingestion, caching, synchronization, and adaptive placement. This repository implements its core algorithmic ideas in a functional SQL backend and a logical hybrid-execution evaluation environment.

| Aspect | Released implementation | Scope of evidence |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Data ingestion and SQL operations | SQLite-backed local/cloud data operations | Functional software behavior |
| Edge–cloud scheduling, caching and replication | Stateful logical hybrid environment | Model-generated behavior |
| Latency, cost, throughput and SLA evaluations | Logical experiment engine | Simulated measurements, not physically measured distributed-system benchmarks |
| Statistical analysis | Matched logical runs; paired Wilcoxon tests and Holm correction | Inference about the implemented model under its assumptions |
| Physical distributed infrastructure | No deployed Spark, Delta Lake, MinIO or multi-node Docker testbed in this software release | Not reproduced by this repository |

**Reproducibility boundary:** The source code does not establish that the performance values reported in the manuscript were measured on a physically deployed distributed testbed. Reproducing any infrastructure-level results requires separately documented infrastructure, workload configurations, execution logs, and matched run-level observations. The repository's logical experiment outputs must not be substituted for independently measured physical-testbed results.

## Key Features

- Hybrid cloud–edge workload execution modelling.

- DQN-based execution-policy learning.

- Predictive latency and cost estimation using multilayer perceptrons (MLPs).

- Feasibility-aware execution placement and action masking.

- Bandwidth- and RTT-aware network-transfer modelling.

- Stateful cache, replication, and resource-queue modelling.

- SLA-aware decisions and cost–latency evaluation.

- SQLite-based transactional SQL storage and synchronization.

- Dataset preparation adapters for public data sources.

- Matched baseline comparisons, ablation experiments, and run-level outputs.

- Paired Wilcoxon signed-rank tests with Holm multiple-comparison adjustment.

- Automated functional testing and high-resolution figure generation.

## Framework Architecture

### 1. Functional SQL Data Path

The storage module performs actual SQLite operations, including transactional ingestion, ACID transaction support, schema and input validation, duplicate detection, edge-side buffering, idempotent synchronization commits, local/cloud SQL reads, and query correctness checks. The use of separate logical edge/cloud components does not itself constitute a physical distributed database deployment.

### 2. Logical Hybrid Execution Engine

The experiment engine provides stateful, sequential, event-driven simulation of hybrid edge/cloud resource availability, placement choices, queue contention, network delays, caching and replication state, SLA decisions, and modelled execution outcomes.

**Scientific implementation note:** Simulated latency, cost, and throughput are generated by the logical execution model. They should not be represented as performance measurements collected from Spark, Delta Lake, MinIO, or a physically distributed cloud–edge infrastructure.

## Algorithmic Components

### Algorithm 1: Edge Data Ingestion and Synchronization

Implemented principally through storage.EdgeAgent, with logical resource and network effects modelled in core.HybridEnvironment. The storage module performs ingestion, buffering, and synchronization operations; the experiment engine models transfer effects.

### Algorithm 2: Predictive Resource Estimation and DQN-Based Decision-Making

The learning module provides two MLP regressors for performance prediction and a DQN execution policy. The documented network configurations use hidden layers of 128 and 64 neurons.

DQN configuration:

| Parameter | Configuration |
| ----------------------- | ---------------------------- |
| Hidden layers | 128, 64 |
| Exploration | Epsilon-based |
| Target-network update | Every 200 gradient updates |
| Replay capacity | 50,000 transitions |
| Discount factor | 0.95 |
| Batch size | 64 |
| Optimizer | Adam |
| DQN learning rate | 0.0001 |
| Stability | Gradient clipping |

Feasible-action selection is handled in the evaluation workflow. Model-specific configurations and checkpoint behavior should be checked against the source code when altering experiments.

### Algorithm 3: Hybrid Execution and Resource Optimization

Implemented through core.HybridEnvironment.step, which models placement, edge/cloud computation, caching, replication, network-transfer costs, resource contention, and SLA outcomes. Actual SQL operations are independently supported through storage.SQLLakehouse.

## Supported Datasets

The repository provides preparation adapters for three public datasets:

| Dataset | Category | Intended use |
| --------------------- | ------------ | ---------------------------- |
| NYC Taxi Trip Records | Batch | Batch-workload preparation |
| DEBS 2014 Smart Home | Streaming | Smart-home event preparation |
| GDELT | Event-driven | Event-oriented preparation |

### Official Data Sources

- [NYC Taxi Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

- [DEBS 2014 Grand Challenge](https://debs.org/grand-challenges/2014/)

- [GDELT Project](https://www.gdeltproject.org/data.html)

Original dataset files are not bundled; obtain them separately and comply with their terms. The prepare command transforms dataset records, whereas the existing evaluate command generates logical workload descriptors and does **not** execute its benchmark directly on those prepared records. Do not describe simulator results as dataset-driven physical benchmark measurements.

## Project Structure

```text
LakehouseFusionAI/
 ├── main.py
 ├── requirements.txt
 ├── lakehousefusion/
 │   ├── __init__.py
 │   ├── core.py
 │   ├── learning.py
 │   ├── evaluation.py
 │   ├── storage.py
 │   ├── datasets.py
 │   └── plots.py
 ├── tests/
 ├── data/
 ├── results/
 └── checkpoints/
```

### Core Modules

| Module | Responsibility |
| ------------- | --------------------------------------------------------------------- |
| core.py | Hybrid environment, action masks, queue states and reward modelling |
| learning.py | MLP regressors and DQN |
| evaluation.py | Baseline comparisons, ablation, summaries and statistical analysis |
| storage.py | SQLite storage, ingestion, buffering and synchronization |
| datasets.py | CSV/Parquet preparation adapters |
| plots.py | Scientific plots |
| main.py | Command-line interface |
| tests/ | Functional tests |

## Installation and Requirements

**Recommended Python version:** 3.11.

### 1. Clone the Repository

```bash
git clone https\://github.com/datasph/LakehouseFusionAI.git\
```
&#x20;cd LakehouseFusionAI

If the source files are inside a nested LakehouseFusionAI/ directory in your checkout, enter that directory before executing main.py.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows Command Prompt:**

```bash
.venv\Scripts\activate
```

**Windows PowerShell:**

.\\.venv\Scripts\Activate.ps1

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

For optional Parquet input/output support:

```bash
python -m pip install pyarrow
```

## Execution Guide

Run the following commands from the directory containing main.py.

### 1. Functional SQL Demonstration

```bash
python main.py sql-demo
```

Demonstrates the SQLite-backed data path and its operations; this is not a distributed Spark deployment.

### 2. Train the Learning Models

```bash
python main.py train --n 350 --epochs 3
```

For a short smoke test:

```bash
python main.py train --n 100 --epochs 1
```

The documented training seed is 42; evaluation seeds begin at 10001. These refer to the repository's software workflow, not necessarily to seed identifiers used to generate manuscript benchmark tables.

### 3. Normal-Network Logical Evaluation

```bash
python main.py evaluate --n 200 --runs 10 --qps 50 --network normal
```

### 4. Constrained-Network Logical Evaluation and Ablation

```bash
python main.py evaluate --n 200 --runs 10 --qps 200 --network constrained --out results/stress --ablation
```

The evaluation command is documented to produce:

- run_metrics.csv — run-level modelled outcomes.

- query_traces.csv — query-level logical traces.

- summary.csv — aggregated results.

- statistics.csv — statistical comparisons.

- metadata.json — experiment metadata.

Inspect the generated output directories for each run; the two evaluation commands above may use different output destinations.

### 5. Prepare Public Data

Obtain suitable CSV input files from the official dataset sources before executing:

**NYC Taxi**

```bash
python main.py prepare --dataset taxi --input data/raw/taxi.csv --out data/processed/taxi.csv
```

**DEBS 2014**

```bash
python main.py prepare --dataset debs --input data/raw/debs.csv --out data/processed/debs.csv
```

**GDELT**

```bash
python main.py prepare --dataset gdelt --input data/raw/gdelt.csv --out data/processed/gdelt.csv
```

Prepared data outputs are separate from the logical evaluation workload generator.

### 6. Generate Figures

```bash
python main.py plot --metrics results/run_metrics.csv --out results/figures
```

This command is documented to generate five 600-DPI PNG plots of **modelled** outcomes. Ensure the metrics path points to an existing evaluation output.

### 7. Run Functional Tests

```bash
python -m pytest tests -q
```

## Experimental Evaluation and Statistics

The evaluation module supports cloud-only, edge-only, static-hybrid and LakehouseFusionAI comparisons under matched logical workload traces. It reports run-level metrics, query-level traces, latency, cost, throughput, SLA outcomes, baseline comparisons, and ablations.

The documented statistical workflow applies two-sided paired Wilcoxon signed-rank tests with Holm adjustment within each metric and condition. The experimental run, not each query in the same run, is the appropriate unit for paired run-level inference. The meaning of each test depends on how runs and matching seeds were generated; review metadata.json, run_metrics.csv, and statistics.csv together.

**Statistical interpretation:** Significant differences in logically simulated outcomes establish differences within the simulation and its assumptions. They do not, by themselves, validate performance improvements on real multi-node infrastructure. The repository's statistical output is not a substitute for paired measurements underlying any manuscript-specific Wilcoxon table unless the same data were actually used.

Offered workload rate (QPS) and completed throughput (QPS) are distinct quantities in the sequential event-driven model.

## Reproducibility

The software supports reproducible algorithm-level research through controlled seeds, separated training/evaluation workflows, matched logical query traces, checkpoints, CSV output, JSON metadata, tests, and figures.

### Reproducibility Boundary and Manuscript Benchmark Alignment

The following distinctions must be preserved when interpreting this release:

1. **Functional reproducibility:** SQLite ingestion, buffering, SQL and synchronization functions can be executed with the released code.

2. **Algorithmic reproducibility:** The predictive models and DQN policy can be trained and evaluated under the implemented logical environment.

3. **Simulation reproducibility:** Modelled latency, costs, SLA results, throughput, and paired statistical comparisons can be regenerated using corresponding settings and seeds.

4. **Infrastructure-level reproducibility:** Physical multi-node Spark/Delta Lake/MinIO/Docker experiments and their measured numerical outputs are **not supplied or reproduced by this release**.

5. **Manuscript numerical comparisons:** The specific manuscript tables and figure values must be traced to their original measurement files or original simulation runs. The repository README does not certify that its generated metrics are identical to those reported in the manuscript.

Reproducing independently obtained infrastructure results requires the original deployment configuration, workload traces, measurement definitions, execution logs, run identifiers, paired observations, and scripts used to calculate tables and figures. If such materials are released later, they should be versioned and cited separately.

## Implementation Scope and Limitations

- SQLite is used for the functional SQL data path; it is not a replacement for Spark or a production-scale distributed lakehouse.

- The cloud–edge performance engine is a logical simulation, not a physical multi-node deployment.

- Network latency, bandwidth limitations and resource contention are modelled within software.

- The logical evaluation command does not directly benchmark the full original NYC Taxi, DEBS or GDELT records.

- Cost outputs are model-defined quantities, not provider billing measurements.

- Simulation outputs alone cannot validate production-scale guarantees, real-world scalability or physical SLA compliance.

- The code archive does not demonstrate that manuscript benchmark numbers were obtained from a physical testbed.

These limitations define the current public implementation and should remain consistent with the Methods, Results and Code Availability statements of the associated manuscript.

## Research Context

**Associated research title:** *AI-Driven Data Lakehouse Optimization Reduces Cost and Latency in Hybrid Cloud–Edge Analytics*.

LakehouseFusionAI provides a functional reference implementation for the manuscript's algorithmic concepts and a controlled environment for investigating resource-aware placement and optimization. The paper's experimental claims should be interpreted in accordance with the actual provenance of their measurements.

## Citation and Code Availability

**GitHub:** [https://github.com/datasph/LakehouseFusionAI](https://github.com/datasph/LakehouseFusionAI)

**Archived software release:** LakehouseFusionAI v1.0.0, Zenodo, 30 October 2026.

**Version-specific DOI:** [https://doi.org/10.5281/zenodo.23796788](https://doi.org/10.5281/zenodo.23796788)

Cite the Zenodo software release when using this version of the implementation. When the associated research article is published, cite the article separately for its research findings. A later GitHub README edit does not automatically update the archived Zenodo v1.0.0 snapshot; publish a new archive version if the revised documentation must itself be part of the preserved release.

## Acknowledgements

The implementation uses public dataset sources and established machine-learning and data-processing libraries. Dataset providers and software contributors retain their respective licensing and attribution rights.

## License

This project is licensed under the MIT License.


