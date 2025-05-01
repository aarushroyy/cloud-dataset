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

class Particle:
    def __init__(self, position, velocity, p_best, p_best_score):
        self.position = position  # Cloudlet-to-VM mapping
        self.velocity = velocity  # Velocity for each cloudlet
        self.p_best = p_best      # Personal best position
        self.p_best_score = p_best_score  # Personal best score

class HybridPSOMinMinScheduler:
    def __init__(self, population_size, max_iterations, task_list, vm_list):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.task_list = task_list
        self.vm_list = vm_list
        
        self.particles = []
        self.random = random.Random()
        
        self.g_best = None  # Global best position
        self.g_best_score = float('inf')
        
        # PSO parameters
        self.w = 0.7   # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
    
    def run(self):
        self.initialize_population()
        
        for iter in range(self.max_iterations):
            for particle in self.particles:
                # Update velocity and position for each task
                for i in range(len(self.task_list)):
                    r1 = self.random.random()
                    r2 = self.random.random()
                    
                    # Cognitive and social components
                    cognitive = self.c1 * r1 * (particle.p_best[i] - particle.position[i])
                    social = self.c2 * r2 * (self.g_best[i] - particle.position[i])
                    # Update velocity
                    particle.velocity[i] = self.w * particle.velocity[i] + cognitive + social
                    
                    # Clamp velocity within [-VMs,VMs]
                    particle.velocity[i] = max(-len(self.vm_list), min(len(self.vm_list), particle.velocity[i]))
                    
                    # Update position (discrete PSO: add velocity and round)
                    new_pos = int(round(particle.position[i] + particle.velocity[i]))
                    # Ensure within VM range
                    new_pos = ((new_pos % len(self.vm_list)) + len(self.vm_list)) % len(self.vm_list)
                    particle.position[i] = new_pos
                
                # Evaluate new position
                fitness = self.fitness(particle.position)
                
                # Update personal best
                if fitness < particle.p_best_score:
                    particle.p_best_score = fitness
                    particle.p_best = particle.position.copy()
                
                # Update global best
                if fitness < self.g_best_score:
                    self.g_best_score = fitness
                    self.g_best = particle.position.copy()
            
            # Print progress
            if iter % 10 == 0:
                print(f"Iteration {iter}, Best fitness: {self.g_best_score}")
        
        return self.g_best
    
    def initialize_population(self):
        self.particles.clear()
        
        # Add MinMin solution as one particle (hybridization)
        minmin_solution = self.minmin_assignment()
        minmin_velocity = [0.0] * len(self.task_list)
        minmin_p_best = minmin_solution.copy()
        minmin_p_best_score = self.fitness(minmin_solution)
        self.particles.append(Particle(minmin_solution, minmin_velocity, minmin_p_best, minmin_p_best_score))
        self.g_best = minmin_solution.copy()
        self.g_best_score = minmin_p_best_score
        
        # Randomly initialize the rest
        for i in range(1, self.population_size):
            position = self.generate_random_solution()
            velocity = [self.random.random() * 2 - 1 for _ in range(len(self.task_list))]  # [-1,1]
            p_best_score = self.fitness(position)
            particle = Particle(position, velocity, position.copy(), p_best_score)
            
            if p_best_score < self.g_best_score:
                self.g_best = position.copy()
                self.g_best_score = p_best_score
            
            self.particles.append(particle)
    
    def minmin_assignment(self):
        """Implements the MinMin algorithm for cloudlet-to-VM mapping initialization."""
        num_tasks = len(self.task_list)
        num_vms = len(self.vm_list)
        assignment = [0] * num_tasks
        assigned = [False] * num_tasks
        
        # Estimated completion times for each VM
        vm_ready_times = [0.0] * num_vms
        
        for k in range(num_tasks):
            selected_task = -1
            selected_vm = -1
            min_completion_time = float('inf')
            
            for i in range(num_tasks):
                if assigned[i]:
                    continue
                task = self.task_list[i]
                for j in range(num_vms):
                    vm = self.vm_list[j]
                    exec_time = task.length / vm.mips
                    completion_time = vm_ready_times[j] + exec_time
                    if completion_time < min_completion_time:
                        min_completion_time = completion_time
                        selected_task = i
                        selected_vm = j
            
            # Assign selected task to selected VM
            assignment[selected_task] = selected_vm
            assigned[selected_task] = True
            # Update VM ready time
            vm_ready_times[selected_vm] += self.task_list[selected_task].length / self.vm_list[selected_vm].mips
        
        return assignment
    
    def generate_random_solution(self):
        solution = [0] * len(self.task_list)
        for i in range(len(self.task_list)):
            solution[i] = self.random.randint(0, len(self.vm_list) - 1)
        return solution
    
    def fitness(self, solution):
        """Fitness function: makespan (minimum is better)"""
        # Calculate the completion time for each VM
        vm_times = [0.0] * len(self.vm_list)
        for i in range(len(self.task_list)):
            vm_idx = solution[i]
            exec_time = self.task_list[i].length / self.vm_list[vm_idx].mips
            vm_times[vm_idx] += exec_time
        
        # Makespan: maximum completion time among all VMs
        return max(vm_times)
    
    def print_evaluation_metrics(self):
        print(f"Best Solution (Makespan): {self.g_best_score}")
        
        # Calculate VM loads for the best solution
        vm_load = [0.0] * len(self.vm_list)
        vm_exec_times = [0.0] * len(self.vm_list)
        
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = self.g_best[i]
            vm = self.vm_list[vm_index]
            
            # Calculate execution time
            exec_time = task.length / vm.mips
            vm_load[vm_index] += task.length
            vm_exec_times[vm_index] += exec_time
        
        # Print detailed metrics
        print("\nVM Load Distribution:")
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
        plt.savefig("hybrid_pso_minmin_results.png")
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
    
    # Initialize PSO-MinMin scheduler
    pop_size = 20
    max_iter = 50
    
    print("Initializing Hybrid PSO-MinMin scheduler...")
    start_time = time.time()
    
    scheduler = HybridPSOMinMinScheduler(pop_size, max_iter, tasks, vms)
    
    # Run the scheduler
    print("Running Hybrid PSO-MinMin optimization...")
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