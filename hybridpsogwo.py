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

class HybridPsoGwo:
    def __init__(self, num_particles, max_iter, task_list, vm_list):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.task_list = task_list
        self.vm_list = vm_list
        self.random = random.Random()
        
        # Initialize lists for positions and velocities
        self.pos = [[0.0 for _ in range(len(task_list))] for _ in range(num_particles)]
        self.vel = [[0.0 for _ in range(len(task_list))] for _ in range(num_particles)]
        
        # Initialize personal and global best positions and fitness
        self.p_best_pos = [[0 for _ in range(len(task_list))] for _ in range(num_particles)]
        self.p_best_fit = [float('inf') for _ in range(num_particles)]
        
        self.g_best_pos = [0 for _ in range(len(task_list))]
        self.g_best_fit = float('inf')
    
    def run(self):
        """Run the hybrid PSO-GWO algorithm"""
        # Initialize swarm
        for i in range(self.num_particles):
            for j in range(len(self.task_list)):
                self.pos[i][j] = self.random.randint(0, len(self.vm_list) - 1)
                self.vel[i][j] = self.random.random()
            
            asg = self.assignment(self.pos[i])
            fit = self.fitness(asg)
            self.p_best_fit[i] = fit
            self.p_best_pos[i] = asg[:]
            
            if fit < self.g_best_fit:
                self.g_best_fit = fit
                self.g_best_pos = asg[:]
        
        # Main optimization loop
        for iter in range(self.max_iter):
            a = 2.0 - 2.0 * iter / (self.max_iter - 1)  # linearly decreasing
            
            for i in range(self.num_particles):
                for j in range(len(self.task_list)):
                    # GWO component (using global best as alpha)
                    r1 = self.random.random()
                    r2 = self.random.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D_alpha = abs(C * self.g_best_pos[j] - self.pos[i][j])
                    X1 = self.g_best_pos[j] - A * D_alpha
                    
                    # PSO component
                    w = 0.5
                    c1 = 1.5
                    c2 = 1.5
                    r1 = self.random.random()
                    r2 = self.random.random()
                    
                    self.vel[i][j] = w * self.vel[i][j] + \
                                   c1 * r1 * (self.p_best_pos[i][j] - self.pos[i][j]) + \
                                   c2 * r2 * (X1 - self.pos[i][j])
                    
                    # Update position and clamp
                    self.pos[i][j] += self.vel[i][j]
                    if self.pos[i][j] < 0:
                        self.pos[i][j] = 0
                    elif self.pos[i][j] > len(self.vm_list) - 1:
                        self.pos[i][j] = len(self.vm_list) - 1
                
                # Evaluate
                asg = self.assignment(self.pos[i])
                fit = self.fitness(asg)
                
                if fit < self.p_best_fit[i]:
                    self.p_best_fit[i] = fit
                    self.p_best_pos[i] = asg[:]
                
                if fit < self.g_best_fit:
                    self.g_best_fit = fit
                    self.g_best_pos = asg[:]
            
            # Print progress
            if iter % 10 == 0:
                print(f"Iteration {iter}, Best fitness: {self.g_best_fit}")
        
        return self.g_best_pos
    
    def assignment(self, pos):
        """Convert continuous pos[] to discrete assignments"""
        asg = [0] * len(pos)
        for i in range(len(pos)):
            asg[i] = int(round(pos[i]))
            # Ensure it's within valid VM range
            asg[i] = min(max(0, asg[i]), len(self.vm_list) - 1)
        return asg
    
    def fitness(self, asg):
        """Fitness = makespan (max total exec time on any VM)"""
        vm_time = [0.0] * len(self.vm_list)
        for i in range(len(asg)):
            vm_idx = asg[i]
            task = self.task_list[i]
            vm = self.vm_list[vm_idx]
            exec_time = task.length / vm.mips
            vm_time[vm_idx] += exec_time
        
        makespan = max(vm_time)
        return makespan
    
    def calculate_makespan(self, assignment=None):
        """Calculate the makespan for the given assignment or best solution"""
        if assignment is None:
            assignment = self.g_best_pos
        
        vm_time = [0.0] * len(self.vm_list)
        for i in range(len(assignment)):
            vm_idx = assignment[i]
            task = self.task_list[i]
            vm = self.vm_list[vm_idx]
            exec_time = task.length / vm.mips
            vm_time[vm_idx] += exec_time
        
        return max(vm_time)
    
    def print_evaluation_metrics(self):
        """Print evaluation metrics for the best solution"""
        best_assignment = self.g_best_pos
        
        # Calculate VM loads for the best solution
        vm_load = [0.0] * len(self.vm_list)
        vm_exec_time = [0.0] * len(self.vm_list)
        
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = best_assignment[i]
            vm = self.vm_list[vm_index]
            
            exec_time = task.length / vm.mips
            vm_load[vm_index] += task.length
            vm_exec_time[vm_index] += exec_time
        
        # Print detailed metrics
        print(f"Best Solution Fitness (Makespan): {self.g_best_fit:.2f}")
        
        print("\nVM Load Distribution:")
        for i in range(len(vm_load)):
            print(f"VM {i}: {vm_load[i]:.2f}")
        
        avg_load = sum(vm_load) / len(vm_load)
        max_load = max(vm_load)
        load_balance_variance = sum((load - avg_load)**2 for load in vm_load)
        throughput = len(self.task_list) / (max(vm_exec_time) + 1)
        
        # Calculate makespan (maximum completion time)
        makespan = max(vm_exec_time)
        
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
        plt.savefig("hybrid_pso_gwo_results.png")
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
    
    # Initialize HybridPsoGwo scheduler
    num_particles = 30
    max_iter = 50
    
    print("Initializing Hybrid PSO-GWO scheduler...")
    start_time = time.time()
    
    scheduler = HybridPsoGwo(num_particles, max_iter, tasks, vms)
    
    # Run the scheduler
    print("Running Hybrid PSO-GWO optimization...")
    best_solution = scheduler.run()
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    metrics = scheduler.print_evaluation_metrics()
    
    # Visualize results
    scheduler.visualize_results(metrics)
    
    # Output task assignments
    print("\nTask Assignments (Task ID -> VM ID):")
    for i, vm_id in enumerate(best_solution):
        if i < 10:  # Just show first 10 for brevity
            print(f"Task {tasks[i].task_id} -> VM {vms[vm_id].vm_id}")
    print("...")

if __name__ == "__main__":
    main()