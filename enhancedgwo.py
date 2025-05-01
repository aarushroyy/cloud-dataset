import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Define Task and VM classes
class Task:
    def __init__(self, task_id, length, resource_request):
        self.task_id = task_id
        self.length = length  # Execution length in MI (Million Instructions)
        self.resource_request = resource_request  # CPU resource request

class VM:
    def __init__(self, vm_id, mips, cpu_cap=1.0):
        self.vm_id = vm_id
        self.mips = mips  # Million Instructions Per Second
        self.cpu_cap = cpu_cap  # CPU capacity

class EnhancedGWO:
    def __init__(self, pop_size, max_iter, task_list, vm_list):
        self.population_size = pop_size
        self.max_iterations = max_iter
        self.task_list = task_list
        self.vm_list = vm_list
        self.population = []
        self.random = random.Random()
        
        # Parameters for optimization objectives
        self.enable_load_balancing = True
        self.enable_utilization = True
        self.enable_wait_time = True
        self.enable_throughput = True
        
        # Alpha, Beta, and Delta wolves
        self.alpha_wolf = None
        self.beta_wolf = None
        self.delta_wolf = None
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
    
    def set_parameters(self, load_balancing, utilization, wait_time, throughput):
        self.enable_load_balancing = load_balancing
        self.enable_utilization = utilization
        self.enable_wait_time = wait_time
        self.enable_throughput = throughput
    
    def run(self):
        # Initialize population with random solutions
        self.initialize_population()
        
        # Main loop of GWO algorithm
        for iter in range(self.max_iterations):
            # Update alpha, beta, and delta wolves
            self.update_leader_wolves()
            
            # Calculate a (linearly decreased from 2 to 0)
            a = 2.0 - iter * (2.0 / self.max_iterations)
            
            # Update each wolf position
            for i in range(self.population_size):
                current_wolf = self.population[i].copy()
                new_position = self.update_wolf_position(current_wolf, a)
                self.population[i] = new_position
        
        # Final update of leader wolves to ensure we have the best solution
        self.update_leader_wolves()
        
        return self.alpha_wolf  # Return the best solution
    
    def initialize_population(self):
        self.population = []
        for i in range(self.population_size):
            self.population.append(self.generate_random_solution())
        
        # Initialize leader wolves with worst possible scores
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
    
    def update_leader_wolves(self):
        # Calculate fitness for each wolf and update leader wolves
        for wolf in self.population:
            fitness_score = self.fitness(wolf)
            
            # Update alpha, beta, and delta
            if fitness_score < self.alpha_score:
                self.delta_score = self.beta_score
                self.delta_wolf = self.beta_wolf.copy() if self.beta_wolf is not None else None
                
                self.beta_score = self.alpha_score
                self.beta_wolf = self.alpha_wolf.copy() if self.alpha_wolf is not None else None
                
                self.alpha_score = fitness_score
                self.alpha_wolf = wolf.copy()
            elif fitness_score < self.beta_score:
                self.delta_score = self.beta_score
                self.delta_wolf = self.beta_wolf.copy() if self.beta_wolf is not None else None
                
                self.beta_score = fitness_score
                self.beta_wolf = wolf.copy()
            elif fitness_score < self.delta_score:
                self.delta_score = fitness_score
                self.delta_wolf = wolf.copy()
    
    def update_wolf_position(self, current_wolf, a):
        new_position = [0] * len(current_wolf)
        
        for i in range(len(current_wolf)):
            # Calculate position updates based on alpha, beta, and delta
            r1 = self.random.random()
            r2 = self.random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            
            r1 = self.random.random()
            r2 = self.random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            
            r1 = self.random.random()
            r2 = self.random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            
            # Calculate position update components
            X1 = self.alpha_wolf[i] - A1 * abs(C1 * self.alpha_wolf[i] - current_wolf[i])
            X2 = self.beta_wolf[i] - A2 * abs(C2 * self.beta_wolf[i] - current_wolf[i])
            X3 = self.delta_wolf[i] - A3 * abs(C3 * self.delta_wolf[i] - current_wolf[i])
            
            # Average the position updates and round to nearest integer
            new_pos = (X1 + X2 + X3) / 3.0
            
            # Ensure the new position is valid (within VM list range)
            vm_index = int(round(new_pos)) % len(self.vm_list)
            if vm_index < 0:
                vm_index += len(self.vm_list)
            
            new_position[i] = vm_index
        
        return new_position
    
    def generate_random_solution(self):
        solution = [0] * len(self.task_list)
        for i in range(len(self.task_list)):
            solution[i] = self.random.randint(0, len(self.vm_list) - 1)
        return solution
    
    def fitness(self, solution):
        vm_load = [0.0] * len(self.vm_list)
        
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = solution[i]
            vm_load[vm_index] += task.length
        
        # Calculate metrics
        load_bal = sum((load - sum(vm_load)/len(vm_load))**2 for load in vm_load)
        util = sum(vm_load)
        wait_time = max(vm_load)
        throughput = len(self.task_list) / (wait_time + 1)
        
        score = 0
        if self.enable_load_balancing:
            score += load_bal
        if self.enable_utilization:
            score -= util
        if self.enable_wait_time:
            score += wait_time
        if self.enable_throughput:
            score -= throughput
        
        return score
    
    def print_evaluation_metrics(self):
        # Calculate VM loads for the best solution
        vm_load = [0.0] * len(self.vm_list)
        vm_exec_times = [0.0] * len(self.vm_list)
        
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = self.alpha_wolf[i]
            vm = self.vm_list[vm_index]
            
            # Execution time = length / mips
            exec_time = task.length / vm.mips
            vm_load[vm_index] += task.length
            vm_exec_times[vm_index] += exec_time
        
        # Print detailed metrics
        print("VM Load Distribution:")
        for i in range(len(vm_load)):
            print(f"VM {i}: {vm_load[i]:.2f}")
        
        avg_load = sum(vm_load) / len(vm_load)
        max_load = max(vm_load)
        load_balance_variance = sum((load - avg_load)**2 for load in vm_load)
        throughput = len(self.task_list) / (max(vm_exec_times) + 1)
        
        # Calculate makespan (maximum completion time)
        makespan = max(vm_exec_times)
        
        print(f"Average Load: {avg_load:.2f}")
        print(f"Load Balance Variance: {load_balance_variance:.2f}")
        print(f"Maximum Wait Time: {max_load:.2f}")
        print(f"Throughput: {throughput:.4f}")
        print(f"Makespan: {makespan:.2f}")
        
        return {
            'vm_load': vm_load,
            'avg_load': avg_load,
            'load_balance_variance': load_balance_variance,
            'max_load': max_load,
            'throughput': throughput,
            'makespan': makespan
        }
    
    def visualize_results(self, metrics):
        """Visualize the load distribution and other key metrics"""
        plt.figure(figsize=(12, 8))
        
        # Plot VM load distribution
        plt.subplot(2, 1, 1)
        plt.bar(range(len(self.vm_list)), metrics['vm_load'], color='skyblue')
        plt.axhline(metrics['avg_load'], color='red', linestyle='--', label=f'Avg Load: {metrics["avg_load"]:.2f}')
        plt.title("VM Load Distribution")
        plt.xlabel("VM ID")
        plt.ylabel("Load")
        plt.legend()
        
        # Plot key metrics
        plt.subplot(2, 1, 2)
        metrics_to_plot = ['makespan', 'throughput', 'load_balance_variance']
        values = [metrics['makespan'], metrics['throughput'], metrics['load_balance_variance']]
        plt.bar(metrics_to_plot, values, color=['green', 'blue', 'orange'])
        plt.title("Key Performance Metrics")
        plt.ylabel("Value")
        
        plt.tight_layout()
        plt.savefig("egwo_results.png")
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
    
    # Create VM objects - using unique machine IDs
    vm_ids = df['machine_id'].unique()
    vms = []
    for vm_id in vm_ids[:4]:  # Limit to 4 VMs like in CloudSim examples
        # Assume all VMs have the same capacity for simplicity
        vms.append(VM(vm_id, 1000))  # 1000 MIPS
    
    return tasks, vms

def main():
    # Process CSV data
    print("Processing CSV data...")
    tasks, vms = process_csv_data("archive/borg_traces_data.csv")
    
    print(f"Created {len(tasks)} tasks and {len(vms)} VMs")
    
    # Initialize EGWO scheduler
    pop_size = 20
    max_iter = 50
    
    print("Initializing EGWO scheduler...")
    start_time = time.time()
    
    egwo = EnhancedGWO(pop_size, max_iter, tasks, vms)
    egwo.set_parameters(True, True, True, True)
    
    # Run the scheduler
    print("Running EGWO optimization...")
    best_solution = egwo.run()
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    metrics = egwo.print_evaluation_metrics()
    
    # Visualize results
    egwo.visualize_results(metrics)
    
    # Output task assignments
    print("\nTask Assignments (Task ID -> VM ID):")
    for i, vm_id in enumerate(best_solution):
        if i < 10:  # Just show first 10 for brevity
            print(f"Task {tasks[i].task_id} -> VM {vms[vm_id].vm_id}")
    print("...")

if __name__ == "__main__":
    main()