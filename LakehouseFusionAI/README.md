# LakehouseFusionAI — Functional Research Reference Implementation

A reproducible open-source-style implementation aligned to *AI-Driven Data Lakehouse Optimization Reduces Cost and Latency in Hybrid Cloud–Edge Analytics*.

## Scope and scientific honesty

**Two distinct backends are present:**
1. **Portable real SQL data path** (`lakehousefusion.storage`) implements actual SQLite ACID transactions, data validation, deduplicated ingestion, edge buffers, idempotent commits, synchronization, local/cloud SQL reads, and query correctness tests. It runs with standard Python plus pandas.
2. **Stateful logical experiment engine** (`lakehousefusion.core`) implements five hybrid execution alternatives, resource queues, bandwidth/RTT transfer modelling, cache and replica transitions, SLA decisions, costs and work traces. **This is an emulation, not real Spark/Delta/MinIO execution.** Performance numbers from this model must not be presented as real cloud-edge benchmark measurements. It is useful for algorithm development and methodology validation.

The manuscript specifies a Docker/Spark 3.5/Delta Lake 3.2/MinIO 4-edge + 1-cloud testbed. That physical testbed and accurate QPS benchmarks require access to such an infrastructure and are **not included**. The Python package is functional but **is not a verified reproduction of the paper's numerical results**. The source does not invent data downloads or claim it used external datasets without files being supplied.

## Quick start (Python 3.11)

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py sql-demo
python main.py train --n 350 --epochs 3
python main.py evaluate --n 200 --runs 10 --qps 50 --network normal
python main.py evaluate --n 200 --runs 10 --qps 200 --network constrained --out results/stress --ablation
python -m pytest tests -q
```

A short training run may be accelerated via `--n 100 --epochs 1`. The training/evaluation seeds are isolated (training 42 and evaluation 10001 onward).

## Genuine dataset adapters

Three public datasets are used by the manuscript: NYC Taxi (batch), DEBS 2014 Smart Home (streaming), GDELT (event). To process downloaded files from these sources:

```
python main.py prepare --dataset taxi --input data/raw/taxi.csv --out data/processed/taxi.csv
python main.py prepare --dataset debs --input data/raw/debs.csv --out data/processed/debs.csv
python main.py prepare --dataset gdelt --input data/raw/gdelt.csv --out data/processed/gdelt.csv
```

Pandas adapters handle common column aliases and fail with an explanatory error if mandatory source fields are absent. DEBS exports have variable schemas; you may need to rename columns according to your downloaded release. For Parquet inputs/outputs install `pyarrow`. `prepare` is an actual dataset transformation; the CLI `evaluate` uses **deterministic generated logical workload descriptors** instead of reading preprocessed dataset records. These are distinct tasks and not claimed to be integrated end-to-end.

Official dataset sources (validate particular releases and licenses yourself):
- NYC TLC: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- GDELT: https://www.gdeltproject.org/data.html
- DEBS 2014 challenge: https://debs.org/grand-challenges/

## Algorithms

- **Algorithm 1**: `storage.EdgeAgent` for real ingestion, buffering and synchronization, and `core.HybridEnvironment` for logical resource/transfer effects. TTL, scheduled transfer and compaction are not implemented in the SQLite backend.
- **Algorithm 2**: `learning.Predictors` two 128/64 MLP regressors; `learning.DQN` 128/64, epsilon exploration, target copy every 200 gradient updates, 50,000 transition replay, gamma 0.95, batch 64, 1e-4 Adam, clipped gradients; `evaluation.plan` performs feasible-action policy selection.
- **Algorithm 3**: `core.HybridEnvironment.step` models placement, local/cloud compute, caching, replication and network costs. `storage.SQLLakehouse` executes actual local SQL independently. Stage-level Spark execution is not implemented.

## Training and evaluation

`train` collects chronological training observations from seeded action exploration, then trains regressors (last 15% of training period used as early-stop validation), and trains DQN in an independent logical environment. No held-out eval data updates policy weights.

`evaluate` emits `run_metrics.csv`, `query_traces.csv`, `summary.csv`, `statistics.csv`, and `metadata.json`. Method comparisons use the same incoming query trace per seed. Tests are run-level, two-sided paired Wilcoxon, Holm correction within each condition/metric. Latency and cost are **emulated**, not measured against external infrastructure. Ten run comparisons produce valid paired tests but statistical significance in an emulated environment is not evidence of production gains.

Current simulator is sequential event-driven and tracks queue contention; not a wall-clock benchmark. Arrival QPS is *offered load*, throughput QPS is *modelled completions divided by modelled elapsed time*. Synthetic time/cost coefficients are illustrative and must be calibrated on actual measurements for paper reproduction.

## Implementation boundaries

- Not an implementation of Delta Lake, Spark, MinIO, Docker orchestration or Linux traffic control.
- No real remote multi-host partition transfer or physically measured bandwidth.
- Cache implementation is stateful LRU metadata with modelled savings; the SQLite data path does not execute cache retrieval from this same model.
- Predictor trains on explored actions, not all counterfactual actions; production would require careful logged-policy off-policy handling.
- `no_edge_preprocess` is a simplifying diagnostic in the logical runner; more advanced ablations require extending plan semantics.
- Stress workloads may be infeasible; numerical outcomes are model-dependent. No alignment to published values is forced.

## Structure

- `core.py`: environment, workloads, states, action mask, resource queues, reward.
- `learning.py`: PyTorch regressors and DQN.
- `evaluation.py`: matched baselines, ablations, summary/statistics.
- `storage.py`: functional SQLite ACID lakehouse, edge ingestion and sync.
- `datasets.py`: authentic dataset CSV/Parquet adapters.
- `main.py`: reproducible command-line interface.
- `tests/`: automated functional tests.

## Figure export
`python main.py plot --metrics results/run_metrics.csv --out results/figures` creates five 600-dpi PNG charts with explicitly labelled modelled outcomes.
