# Cloud Task Scheduling Algorithms

This repository contains Python implementations of various task scheduling algorithms for cloud computing environments. These implementations are based on the CloudSim Plus Java examples but adapted to work with real-world CSV datasets like the Google Borg traces.

## Algorithms Implemented

The following algorithms have been implemented:

1. **Enhanced Grey Wolf Optimizer (EGWO)** - A metaheuristic optimization algorithm inspired by the hunting behavior of grey wolves.
2. **Cooperative Coevolutionary Gene Programming (CCGP)** - A genetic programming approach that coevolves two separate populations for task scheduling.
3. **Hybrid PSO-GWO** - A hybrid of Particle Swarm Optimization and Grey Wolf Optimizer.
4. **Hybrid PSO-MinMin** - A hybrid approach combining Particle Swarm Optimization with the MinMin heuristic.
5. **Modified PSO with Simulated Annealing (MPSOSA)** - Combines PSO with Simulated Annealing for better local search.
6. **Reinforcement Learning Enhanced Grey Wolf Optimizer (RL-GWO)** - Integrates reinforcement learning with GWO.

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn

Install the requirements using:

```bash
pip install pandas numpy matplotlib seaborn
```

## Dataset

The algorithms are designed to work with the Google Borg Traces dataset. You'll need to download the dataset and place it in the `archive` directory:

```
archive/borg_traces_data.csv
```

## Usage

### Running Individual Algorithms

You can run each algorithm independently:

```bash
python egwo_python.py
python ccgp_python.py
python hybrid_pso_gwo.py
python hybrid_pso_minmin.py
python mpsosa_python.py
python rlgwo_python.py
```

Each script will:
1. Load and process the CSV dataset
2. Apply the scheduling algorithm
3. Print performance metrics
4. Visualize the results

### Comparing All Algorithms

To compare all algorithms on the same dataset:

```bash
python algorithm_comparison.py
```

This will:
1. Run all algorithms on the same dataset
2. Generate comparative metrics
3. Create visualizations comparing all algorithms
4. Save results to CSV and image files

## Algorithm Descriptions

### Enhanced Grey Wolf Optimizer (EGWO)

The EGWO algorithm is based on the hunting behavior of grey wolves. It models the social hierarchy and hunting mechanism of grey wolf packs, with alpha, beta, and delta wolves guiding the search. This implementation optimizes the task-to-VM assignments to balance load and minimize makespan.

### Cooperative Coevolutionary Gene Programming (CCGP)

CCGP uses two separate populations that coevolve to solve the scheduling problem. It implements automatically defined functions (ADFs) and uses frequency-based mutation to evolve solutions. This approach is well-suited for complex scheduling problems.

### Hybrid PSO-GWO

This hybrid algorithm combines the strengths of Particle Swarm Optimization (PSO) and Grey Wolf Optimizer (GWO). It uses PSO's ability to explore the search space effectively while leveraging GWO's hierarchical leader-following mechanism for exploitation.

### Hybrid PSO-MinMin

This algorithm combines PSO with the MinMin heuristic to provide better initial solutions. MinMin is a greedy algorithm that assigns the task with the minimum completion time to the VM that can complete it earliest. The PSO component then optimizes this initial solution.

### Modified PSO with Simulated Annealing (MPSOSA)

MPSOSA enhances PSO with Simulated Annealing to escape local optima. It uses a multi-objective approach considering makespan, load balance, resource utilization, and cost. The algorithm includes diversity injection to maintain exploration.

### RL-GWO (Reinforcement Learning Enhanced Grey Wolf Optimizer)

RL-GWO integrates reinforcement learning with GWO to improve exploration and exploitation balance. It maintains a Q-table to learn optimal task-to-VM assignments over time, combining the adaptivity of RL with the optimization capabilities of GWO.

## Output Metrics

All implementations report the following key metrics:

- **Makespan**: Maximum completion time across all VMs (lower is better)
- **Throughput**: Number of tasks completed per unit time (higher is better)
- **Load Balance**: Variance or coefficient of variation of load across VMs (lower is better)
- **VM Utilization**: Distribution of workload across available VMs

## Visualization

Each algorithm generates visualizations showing:

1. VM load distribution
2. Key performance metrics
3. Comparative analysis (when using the comparison tool)

The comparison tool creates additional visualizations:
- Bar charts comparing performance metrics
- Radar charts for multi-metric comparison
- VM load distribution for each algorithm

## Extending the Framework

To add a new algorithm:

1. Create a new Python file with your algorithm implementation
2. Ensure it implements the `run()` and `print_evaluation_metrics()` methods
3. Register it in the `algorithm_comparison.py` file

## Credits

These implementations are based on the CloudSim Plus Java examples, converted and adapted to work with real-world CSV datasets. The original Java implementations were designed for simulation environments.