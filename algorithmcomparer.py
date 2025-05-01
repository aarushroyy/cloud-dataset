import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
import importlib
from collections import defaultdict

# Set the style for better visualizations
plt.style.use('ggplot')
sns.set_palette("muted")

class AlgorithmComparer:
    """Utility for comparing different cloud task scheduling algorithms"""
    
    def __init__(self):
        self.algorithms = {}
        self.results = {}
        self.execution_times = {}
        self.dataset_info = {}
    
    def register_algorithm(self, name, module_name, class_name, **kwargs):
        """Register an algorithm for testing
        
        Args:
            name: Display name for the algorithm
            module_name: Python module name (file name without .py)
            class_name: Class name within the module
            **kwargs: Additional parameters to pass to the algorithm constructor
        """
        self.algorithms[name] = {
            'module_name': module_name,
            'class_name': class_name,
            'kwargs': kwargs
        }
    
    def run_comparison(self, csv_path, max_rows=100, vm_count=4, population_size=20, max_iterations=50):
        """Run all registered algorithms on the same dataset
        
        Args:
            csv_path: Path to the CSV dataset
            max_rows: Maximum number of rows to process
            vm_count: Number of VMs to use
            population_size: Population size for population-based algorithms
            max_iterations: Maximum iterations/generations for iterative algorithms
        """
        # Save dataset info
        self.dataset_info = {
            'csv_path': csv_path,
            'max_rows': max_rows,
            'vm_count': vm_count,
            'population_size': population_size,
            'max_iterations': max_iterations
        }
        
        # Import necessary modules for processing data
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Process data once for all algorithms
        print(f"Processing dataset: {csv_path}")
        df = pd.read_csv(csv_path, nrows=max_rows)
        
        # Get basic dataset statistics
        print(f"Dataset size: {len(df)} rows")
        
        # Run each algorithm
        for alg_name, alg_config in self.algorithms.items():
            print(f"\n{'='*50}")
            print(f"Running {alg_name}...")
            
            # Import the module
            try:
                module = importlib.import_module(alg_config['module_name'])
                # Get the algorithm class
                alg_class = getattr(module, alg_config['class_name'])
                
                # Process data for this algorithm
                tasks, vms = module.process_csv_data(csv_path, max_rows)
                print(f"Created {len(tasks)} tasks and {len(vms)} VMs")
                
                # Create algorithm instance with parameters mapped correctly for each algorithm type
                if alg_config['class_name'] == 'EnhancedGWO':
                    kwargs = {
                        'task_list': tasks,
                        'vm_list': vms,
                        'pop_size': population_size,
                        'max_iter': max_iterations
                    }
                elif alg_config['class_name'] == 'CCGP':
                    kwargs = {
                        'task_list': tasks,
                        'vm_list': vms,
                        'pop_size': population_size,
                        'max_gen': max_iterations,
                        'main_prog_len': 5,
                        'num_adf': 2,
                        'adf_len': 3
                    }
                elif alg_config['class_name'] == 'HybridPsoGwo':
                    kwargs = {
                        'task_list': tasks,
                        'vm_list': vms,
                        'num_particles': population_size,
                        'max_iter': max_iterations
                    }
                else:
                    kwargs = {
                        'task_list': tasks,
                        'vm_list': vms,
                        'population_size': population_size,
                        'max_iterations': max_iterations
                    }
                
                # Add any additional algorithm-specific parameters
                kwargs.update(alg_config['kwargs'])
                
                # Create the instance
                alg_instance = alg_class(**kwargs)
                
                # Run the algorithm and measure time
                start_time = time.time()
                result = alg_instance.run()
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Get metrics
                metrics = alg_instance.print_evaluation_metrics()
                
                # Store results
                self.results[alg_name] = metrics
                self.execution_times[alg_name] = execution_time
                
                print(f"{alg_name} completed in {execution_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error running {alg_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate a comprehensive comparison report of all algorithms"""
        if not self.results:
            print("No results to compare. Run comparison first.")
            return
        
        # Create a DataFrame for comparing key metrics
        metrics_df = pd.DataFrame({
            'Algorithm': list(self.results.keys()),
            'Makespan': [self.results[alg].get('makespan', np.nan) for alg in self.results],
            'Throughput': [self.results[alg].get('throughput', np.nan) for alg in self.results],
            'Execution_Time': [self.execution_times[alg] for alg in self.results]
        })
        
        # Add load balance metrics if available
        if any('load_balance_variance' in self.results[alg] for alg in self.results):
            metrics_df['Load_Balance_Variance'] = [self.results[alg].get('load_balance_variance', np.nan) 
                                                 for alg in self.results]
        
        if any('load_balance_indicator' in self.results[alg] for alg in self.results):
            metrics_df['Load_Balance_Indicator'] = [self.results[alg].get('load_balance_indicator', np.nan) 
                                                  for alg in self.results]
        
        # Calculate VM load coefficient of variation for all algorithms 
        for alg in self.results:
            if 'vm_load' in self.results[alg]:
                vm_loads = self.results[alg]['vm_load']
                mean = np.mean(vm_loads)
                std_dev = np.std(vm_loads)
                cv = std_dev / mean if mean != 0 else 0
                self.results[alg]['Load_Balance_CV'] = cv
        
        metrics_df['Load_Balance_CV'] = [self.results[alg].get('Load_Balance_CV', np.nan) 
                                        for alg in self.results]
        
        # Calculate normalized scores (higher is better)
        # For makespan, lower is better, so invert for normalization
        min_makespan = metrics_df['Makespan'].min()
        metrics_df['Makespan_Score'] = min_makespan / metrics_df['Makespan']
        
        # For throughput, higher is better
        max_throughput = metrics_df['Throughput'].max()
        metrics_df['Throughput_Score'] = metrics_df['Throughput'] / max_throughput
        
        # For load balance CV, lower is better, so invert for normalization
        if 'Load_Balance_CV' in metrics_df.columns:
            metrics_df['Load_Balance_Score'] = 1 / (1 + metrics_df['Load_Balance_CV'])
        
        # For execution time, lower is better
        min_exec_time = metrics_df['Execution_Time'].min()
        metrics_df['Speed_Score'] = min_exec_time / metrics_df['Execution_Time']
        
        # Calculate overall performance score
        score_columns = [col for col in metrics_df.columns if col.endswith('_Score')]
        # metrics_df['Overall_Score'] = metrics_df[score_columns].mean(axis=1)
        
        weights = {
            'Makespan_Score': 0.4,     # Prioritize makespan (40%)
            'Load_Balance_Score': 0.3,  # Load balance is important (30%)
            'Throughput_Score': 0.2,    # Throughput matters (20%)
            'Speed_Score': 0.1          # Algorithm speed less important (10%)
        }

        # Calculate weighted score
        weighted_scores = pd.DataFrame()
        for col in score_columns:
            if col in weights:
                weighted_scores[col] = metrics_df[col] * weights[col]
            else:
                weighted_scores[col] = metrics_df[col] * (1.0 / len(score_columns))
                
        metrics_df['Overall_Score'] = weighted_scores.sum(axis=1)
                
                # Sort by overall score
        metrics_df = metrics_df.sort_values('Overall_Score', ascending=False)
        
        # Print the comparison table
        print("\n" + "="*80)
        print("ALGORITHM COMPARISON RESULTS")
        print("="*80)
        print("\nPerformance Metrics:")
        cols_to_show = ['Algorithm', 'Makespan', 'Throughput', 'Load_Balance_CV', 'Execution_Time', 'Overall_Score']
        cols_to_show = [col for col in cols_to_show if col in metrics_df.columns]
        print(metrics_df[cols_to_show].to_string(index=False))
        
        # Save detailed results to CSV
        metrics_df.to_csv('algorithm_comparison_results.csv', index=False)
        print("\nDetailed results saved to algorithm_comparison_results.csv")
        
        # Create visualizations
        self._visualize_performance_metrics(metrics_df)
        self._visualize_load_distribution()
        self._create_radar_chart(metrics_df)
        self._test_scalability(csv_path=self.dataset_info['csv_path'], algorithm_names=list(self.results.keys()))
        
        return metrics_df
    
    
    
    def _visualize_performance_metrics(self, metrics_df):
        """Create bar charts comparing key performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot Makespan (lower is better)
        ax1 = axes[0, 0]
        sorted_df = metrics_df.sort_values('Makespan')
        sns.barplot(x='Algorithm', y='Makespan', data=sorted_df, ax=ax1)
        ax1.set_title('Makespan (seconds) - Lower is Better', fontsize=14)
        ax1.set_ylabel('Seconds')
        for i, v in enumerate(sorted_df['Makespan']):
            ax1.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        # Plot Throughput (higher is better)
        ax2 = axes[0, 1]
        sorted_df = metrics_df.sort_values('Throughput', ascending=False)
        sns.barplot(x='Algorithm', y='Throughput', data=sorted_df, ax=ax2)
        ax2.set_title('Throughput (tasks/sec) - Higher is Better', fontsize=14)
        ax2.set_ylabel('Tasks/second')
        for i, v in enumerate(sorted_df['Throughput']):
            ax2.text(i, v + 0.005, f"{v:.4f}", ha='center')
        
        # Plot Load Balance CV (lower is better)
        ax3 = axes[1, 0]
        if 'Load_Balance_CV' in metrics_df.columns:
            lb_data = metrics_df.dropna(subset=['Load_Balance_CV']).sort_values('Load_Balance_CV')
            sns.barplot(x='Algorithm', y='Load_Balance_CV', data=lb_data, ax=ax3)
            ax3.set_title('Load Balance (CV) - Lower is Better', fontsize=14)
            ax3.set_ylabel('Coefficient of Variation')
            for i, v in enumerate(lb_data['Load_Balance_CV']):
                ax3.text(i, v + 0.01, f"{v:.2f}", ha='center')
        else:
            ax3.text(0.5, 0.5, "Load Balance CV data not available", ha='center', va='center')
            ax3.set_title('Load Balance')
        
        # Plot Overall Score (higher is better)
        ax4 = axes[1, 1]
        sorted_df = metrics_df.sort_values('Overall_Score', ascending=False)
        sns.barplot(x='Algorithm', y='Overall_Score', data=sorted_df, ax=ax4)
        ax4.set_title('Overall Performance Score', fontsize=14)
        ax4.set_ylabel('Score (higher is better)')
        for i, v in enumerate(sorted_df['Overall_Score']):
            ax4.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig('algorithm_performance_comparison.png')
        plt.close()
        print("Performance metrics visualization saved to algorithm_performance_comparison.png")
    
    def _visualize_load_distribution(self):
        """Create bar charts showing VM load distribution for each algorithm"""
        # Filter algorithms with VM_Loads data
        algs_with_loads = [alg for alg in self.results if 'vm_load' in self.results[alg]]
        
        if not algs_with_loads:
            print("No VM load data available for visualization.")
            return
        
        # Create a figure with subplots
        fig, axes = plt.subplots(len(algs_with_loads), 1, figsize=(12, 3*len(algs_with_loads)))
        
        # Handle case with only one algorithm
        if len(algs_with_loads) == 1:
            axes = [axes]
        
        # Plot each algorithm's VM load distribution
        for i, alg in enumerate(algs_with_loads):
            ax = axes[i]
            loads = self.results[alg]['vm_load']
            
            # Normalize loads for better comparison if values differ greatly
            norm_factor = max(loads) / 100 if max(loads) > 100 else 1
            normalized_loads = [load / norm_factor for load in loads]
            
            # Plot the loads
            bars = ax.bar(range(len(loads)), normalized_loads)
            
            # Add the actual values as text
            for j, v in enumerate(loads):
                ax.text(j, normalized_loads[j] + 1, f"{v:.0f}", ha='center')
            
            # Add mean line
            mean_load = np.mean(loads)
            ax.axhline(mean_load / norm_factor, color='red', linestyle='--')
            
            # Calculate and display CV
            cv = np.std(loads) / mean_load if mean_load > 0 else 0
            ax.set_title(f"{alg} VM Load Distribution (CV: {cv:.2f})", fontsize=12)
            ax.set_xlabel("VM ID")
            ax.set_ylabel(f"Load (÷{norm_factor})" if norm_factor != 1 else "Load")
            ax.set_xticks(range(len(loads)))
        
        plt.tight_layout()
        plt.savefig('vm_load_distribution.png')
        plt.close()
        print("VM load distribution visualization saved to vm_load_distribution.png")
    
    def _create_radar_chart(self, metrics_df):
        """Create a radar chart for multi-metric comparison"""
        # Select metrics for comparison
        metrics = ['Makespan_Score', 'Throughput_Score', 'Load_Balance_Score', 'Speed_Score']
        metrics = [m for m in metrics if m in metrics_df.columns]
        
        if len(metrics) < 3:
            print("Not enough metrics for radar chart.")
            return
        
        # Only include algorithms with data for all metrics
        complete_data = metrics_df.dropna(subset=metrics)
        
        if len(complete_data) < 2:
            print("Not enough algorithms with complete data for radar chart.")
            return
        
        # Number of metrics
        N = len(metrics)
        
        # Calculate angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Set labels
        metric_labels = {
            'Makespan_Score': 'Makespan\n(lower is better)',
            'Throughput_Score': 'Throughput\n(higher is better)',
            'Load_Balance_Score': 'Load Balance\n(more even is better)',
            'Speed_Score': 'Speed\n(faster is better)'
        }
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric_labels.get(m, m) for m in metrics])
        
        # Draw circles and set yticks
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
        ax.set_ylim(0, 1)
        
        # Plot each algorithm
        for i, alg in enumerate(complete_data['Algorithm']):
            values = complete_data.loc[complete_data['Algorithm'] == alg, metrics].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot the algorithm
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=alg)
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Algorithm Performance Comparison', size=15)
        
        plt.savefig('algorithm_radar_chart.png')
        plt.close()
        print("Radar chart saved to algorithm_radar_chart.png")
        
    def _test_scalability(self, csv_path, algorithm_names=None):
        """Test how algorithms scale with increasing dataset size"""
        if algorithm_names is None:
            algorithm_names = list(self.algorithms.keys())
            
        sizes = [100, 500, 1000, 1500, 2000]
        results = {alg: {'sizes': sizes, 'makespan': [], 'time': []} for alg in algorithm_names}
        
        for size in sizes:
            for alg_name in algorithm_names:
                # Run with this dataset size
                metrics = self.run_single_algorithm(alg_name, csv_path, size)
                results[alg_name]['makespan'].append(metrics['makespan'])
                results[alg_name]['time'].append(metrics['execution_time'])
        
        # Plot scaling results
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        for alg in algorithm_names:
            plt.plot(sizes, results[alg]['makespan'], marker='o', label=alg)
        plt.title('Makespan Scaling with Dataset Size')
        plt.xlabel('Number of Tasks')
        plt.ylabel('Makespan')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for alg in algorithm_names:
            plt.plot(sizes, results[alg]['time'], marker='o', label=alg)
        plt.title('Algorithm Execution Time Scaling')
        plt.xlabel('Number of Tasks') 
        plt.ylabel('Execution Time (s)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('algorithm_scaling.png')

def main():
    """Main function to run comparison of all algorithms"""
    comparer = AlgorithmComparer()
    
    # Register all algorithms with correct module names
    comparer.register_algorithm("EGWO", "enhancedgwo", "EnhancedGWO")
    comparer.register_algorithm("CCGP", "ccgp", "CCGP")
    comparer.register_algorithm("HybridPSOGWO", "hybridpsogwo", "HybridPsoGwo")
    comparer.register_algorithm("HybridPSOMinMin", "hybridpsominmin", "HybridPSOMinMinScheduler")
    comparer.register_algorithm("MPSOSA", "mpsosa", "MPSOSA")
    comparer.register_algorithm("RLGWO", "rlgwo", "RLGWO")
    
    # Run comparison
    comparer.run_comparison(
        csv_path="archive/borg_traces_data.csv",
        max_rows=800,
        vm_count=4,
        population_size=20,
        max_iterations=50
    )

if __name__ == "__main__":
    main()