import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
getcontext().prec = 28  # Set precision for decimal calculations

# Define Task and VM classes
class Task:
    def __init__(self, task_id, length, resource_request):
        self.task_id = task_id
        self.length = length  # Execution length in MI (Million Instructions)
        self.resource_request = resource_request  # CPU resource request

class VM:
    def __init__(self, vm_id, mips, pes=1):
        self.vm_id = vm_id
        self.mips = mips  # Million Instructions Per Second
        self.pes = pes    # Number of processing elements

class RLGWO:
    """RL-GWO Hybrid Task Scheduling Algorithm"""
    def __init__(self, task_list, vm_list, population_size=20, max_iterations=50):
        self.task_list = task_list
        self.vm_list = vm_list
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.random = random.Random()
        
        # RL parameters
        self.alpha = 0.1      # Learning rate
        self.gamma = 0.9      # Discount factor
        self.epsilon = 0.3    # Exploration rate
        
        # Initialize Q-table: [task][vm]
        self.q_table = [[0.0 for _ in range(len(vm_list))] for _ in range(len(task_list))]
        
        # Best solution found
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def run(self):
        """Run the RL-GWO hybrid scheduling algorithm"""
        # Initialize wolf population (each wolf is a potential solution/mapping)
        wolves = []
        for i in range(self.population_size):
            wolves.append(self.initialize_random_solution())
        
        # Initialize alpha, beta, and delta wolves
        alpha_wolf = None
        beta_wolf = None
        delta_wolf = None
        alpha_score = float('inf')
        beta_score = float('inf')
        delta_score = float('inf')
        
        # Start iterations
        for iter in range(self.max_iterations):
            # Update a parameter (linearly decreasing from 2 to 0)
            a = 2.0 - 2.0 * iter / (self.max_iterations - 1)
            
            # Evaluate each wolf
            for i in range(self.population_size):
                current_wolf = wolves[i]
                fitness = self.calculate_fitness(current_wolf)
                
                # Update alpha, beta, delta wolves
                if fitness < alpha_score:
                    delta_score = beta_score
                    delta_wolf = beta_wolf.copy() if beta_wolf is not None else None
                    beta_score = alpha_score
                    beta_wolf = alpha_wolf.copy() if alpha_wolf is not None else None
                    alpha_score = fitness
                    alpha_wolf = current_wolf.copy()
                elif fitness < beta_score:
                    delta_score = beta_score
                    delta_wolf = beta_wolf.copy() if beta_wolf is not None else None
                    beta_score = fitness
                    beta_wolf = current_wolf.copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_wolf = current_wolf.copy()
            
            # Update each wolf's position using both GWO and RL
            for i in range(self.population_size):
                current_wolf = wolves[i].copy()
                new_position = {}
                
                # For each task assignment
                for task_id in range(len(self.task_list)):
                    # Use epsilon-greedy strategy for exploration vs exploitation
                    if self.random.random() < self.epsilon:
                        # Explore: Choose random VM
                        new_position[task_id] = self.random.randint(0, len(self.vm_list) - 1)
                    else:
                        # Exploit: Use GWO to update position based on alpha, beta, delta
                        
                        # Get current positions of pack leaders for this task
                        alpha_pos = alpha_wolf[task_id] if alpha_wolf is not None else 0
                        beta_pos = beta_wolf[task_id] if beta_wolf is not None else 0
                        delta_pos = delta_wolf[task_id] if delta_wolf is not None else 0
                        
                        # Generate random components for position update
                        r1 = self.random.random()
                        r2 = self.random.random()
                        
                        # Calculate position updates toward each leader
                        a1 = 2 * a * r1 - a
                        a2 = 2 * a * r2 - a
                        a3 = 2 * a * self.random.random() - a
                        
                        c1 = 2 * self.random.random()
                        c2 = 2 * self.random.random()
                        c3 = 2 * self.random.random()
                        
                        # Calculate direction vectors toward leaders
                        d_alpha = abs(c1 * alpha_pos - current_wolf[task_id])
                        d_beta = abs(c2 * beta_pos - current_wolf[task_id])
                        d_delta = abs(c3 * delta_pos - current_wolf[task_id])
                        
                        # Calculate step sizes
                        x1 = alpha_pos - a1 * d_alpha
                        x2 = beta_pos - a2 * d_beta
                        x3 = delta_pos - a3 * d_delta
                        
                        # Combine influences (discretized for VM assignment)
                        combined_influence = (x1 + x2 + x3) / 3.0
                        new_vm_id = int(round(combined_influence)) % len(self.vm_list)
                        if new_vm_id < 0:
                            new_vm_id += len(self.vm_list)
                        
                        # Apply reinforcement learning: update Q-table
                        current_vm_id = current_wolf[task_id]
                        reward = -self.calculate_task_cost_on_vm(self.task_list[task_id], self.vm_list[new_vm_id])
                        
                        # Q-Learning update
                        max_future_q = self.get_max_q_value(task_id)
                        self.q_table[task_id][new_vm_id] = (1 - self.alpha) * self.q_table[task_id][current_vm_id] + \
                                                          self.alpha * (reward + self.gamma * max_future_q)
                        
                        # Use the new position
                        new_position[task_id] = new_vm_id
                
                # Update wolf position
                wolves[i] = new_position
            
            # Reduce exploration rate gradually
            self.epsilon = max(0.05, self.epsilon * 0.99)
            
            # Print progress
            if iter % 10 == 0:
                print(f"Iteration {iter}, Best fitness: {alpha_score}")
        
        # Store the best solution
        self.best_solution = alpha_wolf
        self.best_fitness = alpha_score
        
        return alpha_wolf
    
    def get_max_q_value(self, task_id):
        """Get maximum Q-value for a task across all VMs"""
        max_q = float('-inf')
        for j in range(len(self.vm_list)):
            if self.q_table[task_id][j] > max_q:
                max_q = self.q_table[task_id][j]
        return max_q
    
    def initialize_random_solution(self):
        """Initialize random solution (wolf position)"""
        solution = {}
        for i in range(len(self.task_list)):
            solution[i] = self.random.randint(0, len(self.vm_list) - 1)
        return solution
    
    def calculate_fitness(self, solution):
        """Calculate fitness for wolf position (lower is better - we use makespan and load balance)"""
        # Calculate load on each VM
        vm_load = [0.0] * len(self.vm_list)
        vm_exec_time = [0.0] * len(self.vm_list)
        
        for task_id in range(len(self.task_list)):
            vm_id = solution[task_id]
            task = self.task_list[task_id]
            vm = self.vm_list[vm_id]
            
            # Get PE count
            pe_count = vm.pes
            exec_time = task.length / (vm.mips * pe_count)
            vm_load[vm_id] += task.length
            vm_exec_time[vm_id] += exec_time
        
        # Calculate makespan (maximum execution time across all VMs)
        makespan = max(vm_exec_time)
        
        # Calculate standard deviation of load (for load balancing)
        avg_load = sum(vm_load) / len(vm_load)
        load_variance = sum((load - avg_load)**2 for load in vm_load) / len(vm_load)
        load_std_dev = np.sqrt(load_variance)
        
        # Fitness is a combination of makespan and load balance
        return makespan * 0.7 + load_std_dev * 0.3
    
    def calculate_task_cost_on_vm(self, task, vm):
        """Calculate the cost of running a task on a VM"""
        # Get PE count
        pe_count = vm.pes
        exec_time = task.length / (vm.mips * pe_count)
        cost = exec_time * vm.mips * 0.01  # Arbitrary cost model
        return cost
    
    def calculate_makespan(self):
        """Calculate the makespan for the best solution"""
        if self.best_solution is None:
            return 0.0
        
        vm_exec_time = [0.0] * len(self.vm_list)
        for task_id in range(len(self.task_list)):
            vm_id = self.best_solution[task_id]
            task = self.task_list[task_id]
            vm = self.vm_list[vm_id]
            
            # Get PE count
            pe_count = vm.pes
            exec_time = task.length / (vm.mips * pe_count)
            vm_exec_time[vm_id] += exec_time
        
        return max(vm_exec_time)
    
    def print_evaluation_metrics(self):
        """Print evaluation metrics for the best solution"""
        if self.best_solution is None:
            print("No solution found.")
            return {}
        
        vm_load = [0.0] * len(self.vm_list)
        vm_exec_time = [0.0] * len(self.vm_list)
        vm_utilization = [0.0] * len(self.vm_list)
        
        for task_id in range(len(self.task_list)):
            vm_id = self.best_solution[task_id]
            task = self.task_list[task_id]
            vm = self.vm_list[vm_id]
            
            # Get PE count
            pe_count = vm.pes
            exec_time = task.length / (vm.mips * pe_count)
            vm_load[vm_id] += task.length
            vm_exec_time[vm_id] += exec_time
            vm_utilization[vm_id] += exec_time  # Simplified utilization model
        
        print("\n========== CUSTOM METRICS ==========")
        
        makespan = max(vm_exec_time)
        total_execution_time = sum(vm_exec_time)
        
        print(f"Total execution time: {total_execution_time:.2f} seconds")
        print(f"Makespan: {makespan:.2f} seconds")
        
        # Calculate VM utilization
        print("\nVM Utilization:")
        for i in range(len(vm_utilization)):
            print(f"VM {i}: {vm_utilization[i]:.2f} seconds")
        
        # Calculate standard deviation for load balancing
        avg_utilization = sum(vm_utilization) / len(vm_utilization)
        variance = sum((util - avg_utilization)**2 for util in vm_utilization) / len(vm_utilization)
        std_dev = np.sqrt(variance)
        
        print(f"\nAverage VM utilization: {avg_utilization:.2f} seconds")
        print(f"VM utilization Std Dev: {std_dev:.2f} seconds")
        print(f"Load balance indicator: {(std_dev / avg_utilization):.2f}")
        
        # Calculate throughput
        throughput = len(self.task_list) / makespan
        print(f"Throughput: {throughput:.4f} tasks/s")
        
        return {
            'makespan': makespan,
            'total_execution_time': total_execution_time,
            'vm_utilization': vm_utilization,
            'avg_utilization': avg_utilization,
            'utilization_std_dev': std_dev,
            'load_balance_indicator': std_dev / avg_utilization,
            'throughput': throughput,
            'vm_load': vm_load
        }
    
    def visualize_results(self, metrics):
        """Visualize the results of the scheduling"""
        plt.figure(figsize=(15, 10))
        
        # Plot VM load distribution
        plt.subplot(2, 2, 1)
        plt.bar(range(len(self.vm_list)), metrics['vm_load'], color='skyblue')
        plt.axhline(sum(metrics['vm_load']) / len(metrics['vm_load']), color='red', linestyle='--', 
                   label=f"Avg Load: {sum(metrics['vm_load']) / len(metrics['vm_load']):.2f}")
        plt.title("VM Load Distribution")
        plt.xlabel("VM ID")
        plt.ylabel("Load")
        plt.legend()
        
        # Plot VM utilization
        plt.subplot(2, 2, 2)
        plt.bar(range(len(self.vm_list)), metrics['vm_utilization'], color='green')
        plt.axhline(metrics['avg_utilization'], color='red', linestyle='--', 
                   label=f"Avg Utilization: {metrics['avg_utilization']:.2f}")
        plt.title("VM Utilization")
        plt.xlabel("VM ID")
        plt.ylabel("Utilization (seconds)")
        plt.legend()
        
        # Plot performance metrics
        plt.subplot(2, 2, 3)
        perf_metrics = ['makespan', 'throughput', 'load_balance_indicator']
        perf_values = [metrics['makespan'], metrics['throughput'], metrics['load_balance_indicator']]
        plt.bar(perf_metrics, perf_values, color=['blue', 'orange', 'purple'])
        plt.title("Performance Metrics")
        plt.ylabel("Value")
        
        # Plot RL-GWO convergence (if tracking was implemented)
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, "RL-GWO combines the exploration capabilities of\nReinforcement Learning with the optimization\npower of Grey Wolf Optimizer",
                horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title("Algorithm Info")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("rlgwo_results.png")
        plt.show()

def process_csv_data(file_path, max_rows=100):
    """Process CSV data and create task and VM objects"""
    df = pd.read_csv(file_path, nrows=max_rows)
    
    # Convert columns to numeric as needed and fill missing values
    df['resource_request'] = pd.to_numeric(df['resource_request'].apply(
        lambda x: eval(x)['cpus'] if isinstance(x, str) and 'cpus' in eval(x) else 0.1
    ), errors='coerce').fillna(0.1)
    
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce').fillna(0)
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce').fillna(1)
    
    # Create Task objects
    tasks = []
    for idx, row in df.iterrows():
        task_id = idx
        # Calculate length based on execution time and a nominal MIPS value
        exec_time_seconds = max((row['end_time'] - row['start_time']) / 1e9, 0.1)  # Convert ns to s, minimum 0.1s
        length = exec_time_seconds * 1000  # Assume 1000 MIPS as base
        resource_req = row['resource_request']
        tasks.append(Task(task_id, length, resource_req))
    
    # Create VM objects with varying PE counts (1-3 PEs like in the Java example)
    vm_ids = df['machine_id'].unique()
    vms = []
    for i, vm_id in enumerate(vm_ids[:4]):  # Limit to 4 VMs like in CloudSim examples
        pes = 1 + (i % 3)  # Varying PE count from 1-3
        vms.append(VM(vm_id, 1000, pes))  # 1000 MIPS
    
    return tasks, vms

def main():
    # Process CSV data
    print("Processing CSV data...")
    tasks, vms = process_csv_data("archive/borg_traces_data.csv")
    
    print(f"Created {len(tasks)} tasks and {len(vms)} VMs")
    for i, vm in enumerate(vms):
        print(f"VM {i}: ID={vm.vm_id}, MIPS={vm.mips}, PEs={vm.pes}")
    
    # Initialize RL-GWO scheduler
    population_size = 20
    max_iterations = 50
    
    print("Initializing RL-GWO scheduler...")
    start_time = time.time()
    
    scheduler = RLGWO(tasks, vms, population_size, max_iterations)
    
    # Run the scheduler
    print("Running RL-GWO optimization...")
    best_solution = scheduler.run()
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print evaluation metrics
    metrics = scheduler.print_evaluation_metrics()
    
    # Visualize results
    scheduler.visualize_results(metrics)
    
    # Output task assignments
    print("\nTask Assignments (Task ID -> VM ID):")
    for i, (task_id, vm_id) in enumerate(best_solution.items()):
        if i < 10:  # Just show first 10 for brevity
            print(f"Task {task_id} -> VM {vms[vm_id].vm_id}")
    print("...")

if __name__ == "__main__":
    main()