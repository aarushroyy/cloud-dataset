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

class MPSOSA:
    def __init__(self, population_size, max_iterations, task_list, vm_list):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.task_list = task_list
        self.vm_list = vm_list
        self.random = random.Random()
        
        # PSO parameters
        self.w_fixed = 0.7    # Inertia weight
        self.c1 = 1.5   # Cognitive coefficient
        self.c2 = 1.5   # Social coefficient
        
        # SA parameters
        self.initial_temp = 100.0
        self.cooling_rate = 0.95
        
        # Objective weights
        self.weight_makespan = 0.4
        self.weight_load_balance = 0.3
        self.weight_utilization = 0.2
        self.weight_cost = 0.1
        
        # Particles and velocities
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        # Metrics for the best solution
        self.best_makespan = 0.0
        self.best_load_balance_score = 0.0
        self.best_utilization_score = 0.0
        self.best_cost = 0.0
    
    def set_objective_weights(self, makespan, load_balance, utilization, cost):
        """Set weights for different optimization objectives"""
        self.weight_makespan = makespan
        self.weight_load_balance = load_balance
        self.weight_utilization = utilization
        self.weight_cost = cost
    
    def get_best_makespan(self):
        return self.best_makespan
    
    def get_best_load_balance_score(self):
        return self.best_load_balance_score
    
    def get_best_utilization_score(self):
        return self.best_utilization_score
    
    def get_best_cost(self):
        return self.best_cost
    
    def run(self):
        """Run the MPSOSA algorithm"""
        # Initialize particles and velocities
        self.initialize_particles()
        
        # For convergence tracking
        previous_best_fitness = float('inf')
        stagnation_counter = 0
        w = self.w_fixed  # Use local variable for w since we'll modify it
        
        # Main PSO loop
        for iter in range(self.max_iterations):
            # Calculate current temperature for SA
            temperature = self.initial_temp * (self.cooling_rate ** iter)
            
            # Update each particle
            for i in range(self.population_size):
                # Update velocity and position
                self.update_particle(i, w)
                
                # Apply simulated annealing for local search
                if self.random.random() < 0.3:  # 30% chance to apply SA
                    self.apply_simulated_annealing(i, temperature)
                
                # Evaluate the new position
                fitness = self.calculate_fitness(self.particles[i])
                
                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_bests[i] = self.particles[i].copy()
                    
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = self.particles[i].copy()
                        
                        # Store metrics for the best solution
                        self.update_best_metrics(self.global_best)
            
            # Dynamic parameter adjustment
            w = 0.9 - (0.5 * iter / self.max_iterations)  # Gradually decrease inertia weight
            
            # Convergence check
            if abs(previous_best_fitness - self.global_best_fitness) < 0.001:
                stagnation_counter += 1
                if stagnation_counter > 5:
                    # Apply diversity injection
                    self.inject_diversity()
                    stagnation_counter = 0
            else:
                stagnation_counter = 0
                previous_best_fitness = self.global_best_fitness
            
            # Print progress periodically
            if iter % 10 == 0:
                print(f"Iteration {iter}, Best fitness: {self.global_best_fitness}")
        
        # Final evaluation of the best solution
        self.update_best_metrics(self.global_best)
        return self.global_best
    
    def initialize_particles(self):
        """Initialize particles with random positions and velocities"""
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        self.personal_best_fitness = []
        
        for i in range(self.population_size):
            # Generate random initial position
            particle = []
            for j in range(len(self.task_list)):
                particle.append(self.random.randint(0, len(self.vm_list) - 1))
            self.particles.append(particle)
            
            # Initialize velocity with small random values
            velocity = []
            for j in range(len(self.task_list)):
                velocity.append((self.random.random() - 0.5) * 2)  # Between -1 and 1
            self.velocities.append(velocity)
            
            # Initialize personal best
            self.personal_bests.append(particle.copy())
            fitness = self.calculate_fitness(particle)
            self.personal_best_fitness.append(fitness)
            
            # Update global best if needed
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = particle.copy()
    
    def update_particle(self, index, w):
        """Update particle velocity and position using PSO equations"""
        particle = self.particles[index]
        velocity = self.velocities[index]
        p_best = self.personal_bests[index]
        
        # Update velocity
        for i in range(len(particle)):
            # PSO velocity formula: v = w*v + c1*r1*(pBest-x) + c2*r2*(gBest-x)
            r1 = self.random.random()
            r2 = self.random.random()
            
            velocity[i] = w * velocity[i] + \
                         self.c1 * r1 * (p_best[i] - particle[i]) + \
                         self.c2 * r2 * (self.global_best[i] - particle[i])
            
            # Clamp velocity to avoid excessive movement
            velocity[i] = max(-len(self.vm_list), min(len(self.vm_list), velocity[i]))
        
        # Update position
        for i in range(len(particle)):
            # New position = old position + velocity (with bounds checking)
            new_pos = particle[i] + round(velocity[i])
            
            # Ensure position is within valid VM indices
            while new_pos < 0:
                new_pos += len(self.vm_list)
            new_pos %= len(self.vm_list)
            
            particle[i] = new_pos
    
    def apply_simulated_annealing(self, particle_index, temperature):
        """Apply simulated annealing for local search"""
        particle = self.particles[particle_index]
        current_fitness = self.calculate_fitness(particle)
        
        # Create a neighbor by swapping a random task assignment
        neighbor = particle.copy()
        task_index = self.random.randint(0, len(self.task_list) - 1)
        new_vm_index = self.random.randint(0, len(self.vm_list) - 1)
        neighbor[task_index] = new_vm_index
        
        # Calculate neighbor fitness
        neighbor_fitness = self.calculate_fitness(neighbor)
        
        # Decide whether to accept the neighbor solution
        if neighbor_fitness < current_fitness:
            # Accept better solution
            self.particles[particle_index] = neighbor
        else:
            # Accept worse solution with probability based on temperature
            acceptance_probability = np.exp((current_fitness - neighbor_fitness) / temperature)
            if self.random.random() < acceptance_probability:
                self.particles[particle_index] = neighbor
    
    def inject_diversity(self):
        """Replace the worst particles with new random solutions for diversity"""
        # Replace the worst 20% of particles with new random solutions
        num_to_replace = max(1, self.population_size // 5)
        
        # Find indices of particles with worst fitness
        indices = list(range(self.population_size))
        indices.sort(key=lambda i: self.personal_best_fitness[i], reverse=True)
        
        # Replace worst particles
        for i in range(num_to_replace):
            idx = indices[i]
            
            # Generate new random solution
            new_particle = []
            for j in range(len(self.task_list)):
                new_particle.append(self.random.randint(0, len(self.vm_list) - 1))
            
            # Reset velocity
            new_velocity = []
            for j in range(len(self.task_list)):
                new_velocity.append((self.random.random() - 0.5) * 2)
            
            # Update particle and velocity
            self.particles[idx] = new_particle
            self.velocities[idx] = new_velocity
            
            # Update personal best
            fitness = self.calculate_fitness(new_particle)
            self.personal_bests[idx] = new_particle.copy()
            self.personal_best_fitness[idx] = fitness
    
    def calculate_fitness(self, solution):
        """Calculate fitness with multiple weighted objectives"""
        # Setup for calculation
        vm_load = [0.0] * len(self.vm_list)
        vm_processing_times = [0.0] * len(self.vm_list)
        vm_costs = [0.0] * len(self.vm_list)
        
        # Calculate load and processing times for each VM
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = solution[i]
            vm = self.vm_list[vm_index]
            
            execution_time = task.length / vm.mips
            vm_load[vm_index] += task.length
            vm_processing_times[vm_index] += execution_time
            
            # Compute cost (simplified model: time * MIPS cost)
            cost_per_mips = 0.01 + (0.005 * vm_index)  # Higher index VMs cost more
            vm_costs[vm_index] += execution_time * cost_per_mips
        
        # Calculate metrics
        makespan = max(vm_processing_times)
        avg_load = sum(vm_load) / len(vm_load)
        load_balance_score = sum((load - avg_load)**2 for load in vm_load) / len(self.vm_list)  # Variance
        utilization = sum(vm_load)
        total_cost = sum(vm_costs)
        
        # Normalize metrics
        norm_makespan = makespan / 1000.0
        norm_load_balance = load_balance_score / 1000000.0
        norm_utilization = 1.0 - (utilization / (10000.0 * len(self.vm_list)))
        norm_cost = total_cost / 100.0
        
        # Combine metrics using weights
        return (self.weight_makespan * norm_makespan) + \
               (self.weight_load_balance * norm_load_balance) + \
               (self.weight_utilization * norm_utilization) + \
               (self.weight_cost * norm_cost)
    
    def update_best_metrics(self, solution):
        """Store metrics for the best solution"""
        vm_load = [0.0] * len(self.vm_list)
        vm_processing_times = [0.0] * len(self.vm_list)
        vm_costs = [0.0] * len(self.vm_list)
        
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = solution[i]
            vm = self.vm_list[vm_index]
            
            execution_time = task.length / vm.mips
            vm_load[vm_index] += task.length
            vm_processing_times[vm_index] += execution_time
            
            cost_per_mips = 0.01 + (0.005 * vm_index)
            vm_costs[vm_index] += execution_time * cost_per_mips
        
        avg_load = sum(vm_load) / len(vm_load)
        self.best_makespan = max(vm_processing_times)
        self.best_load_balance_score = sum((load - avg_load)**2 for load in vm_load) / len(self.vm_list)
        self.best_utilization_score = sum(vm_load) / len(self.vm_list)
        self.best_cost = sum(vm_costs)
    
    def get_vm_load_distribution(self):
        """Get VM load distribution for the best solution"""
        vm_load = [0.0] * len(self.vm_list)
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = self.global_best[i]
            vm_load[vm_index] += task.length
        
        return vm_load
    
    def print_evaluation_metrics(self):
        """Print evaluation metrics for the best solution"""
        print("\nEvaluation Metrics:")
        print(f"Best Makespan: {self.best_makespan:.2f}")
        print(f"Best Load Balance Score: {self.best_load_balance_score:.2f}")
        print(f"Best Utilization Score: {self.best_utilization_score:.2f}")
        print(f"Best Cost: {self.best_cost:.2f}")
        
        # Calculate and display VM load distribution
        vm_load = self.get_vm_load_distribution()
        
        print("\nVM Load Distribution:")
        for i in range(len(vm_load)):
            print(f"VM {i}: {vm_load[i]:.2f}")
        
        avg_load = sum(vm_load) / len(vm_load)
        load_balance_variance = sum((load - avg_load)**2 for load in vm_load)
        
        print(f"Average Load: {avg_load:.2f}")
        print(f"Load Balance Variance: {load_balance_variance:.2f}")
        
        return {
            'makespan': self.best_makespan,
            'load_balance_score': self.best_load_balance_score,
            'utilization_score': self.best_utilization_score,
            'cost': self.best_cost,
            'vm_load': vm_load,
            'avg_load': avg_load,
            'load_balance_variance': load_balance_variance
        }
    
    def visualize_results(self, metrics):
        """Visualize the load distribution and key metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot VM load distribution
        plt.subplot(2, 2, 1)
        vm_labels = [f"VM {i}" for i in range(len(self.vm_list))]
        plt.bar(vm_labels, metrics['vm_load'], color='skyblue')
        plt.axhline(metrics['avg_load'], color='red', linestyle='--', label=f'Avg Load: {metrics["avg_load"]:.2f}')
        plt.title("VM Load Distribution")
        plt.ylabel("Load")
        plt.legend()
        
        # Plot makespan and utilization
        plt.subplot(2, 2, 2)
        metrics_1 = ['makespan', 'utilization_score']
        values_1 = [metrics['makespan'], metrics['utilization_score']]
        plt.bar(metrics_1, values_1, color=['green', 'orange'])
        plt.title("Makespan and Utilization")
        plt.ylabel("Value")
        
        # Plot load balance and cost
        plt.subplot(2, 2, 3)
        metrics_2 = ['load_balance_score', 'cost']
        values_2 = [metrics['load_balance_score'], metrics['cost']]
        plt.bar(metrics_2, values_2, color=['blue', 'purple'])
        plt.title("Load Balance and Cost")
        plt.ylabel("Value")
        
        # Plot proportion of weight contribution
        plt.subplot(2, 2, 4)
        weights = ['Makespan', 'Load Balance', 'Utilization', 'Cost']
        weight_values = [self.weight_makespan, self.weight_load_balance, 
                         self.weight_utilization, self.weight_cost]
        plt.pie(weight_values, labels=weights, autopct='%1.1f%%', 
                startangle=90, colors=['green', 'blue', 'orange', 'purple'])
        plt.title("Objective Weight Distribution")
        
        plt.tight_layout()
        plt.savefig("mpsosa_results.png")
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
    
    # Initialize MPSOSA scheduler
    pop_size = 20
    max_iter = 50
    
    print("Initializing MPSOSA scheduler...")
    start_time = time.time()
    
    scheduler = MPSOSA(pop_size, max_iter, tasks, vms)
    # Set weights for different optimization objectives
    scheduler.set_objective_weights(0.4, 0.3, 0.2, 0.1)
    
    # Run the scheduler
    print("Running MPSOSA optimization...")
    best_solution = scheduler.run()
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print evaluation metrics
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