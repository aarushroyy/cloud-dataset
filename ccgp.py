import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from copy import deepcopy

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

class Chromosome:
    """Chromosome representation for SL-GEP with ADFs"""
    def __init__(self, main_program, adfs):
        self.main_program = main_program  # Main program (priority function for TSR or RSR)
        self.adfs = adfs  # ADFs (Automatically Defined Functions)
    
    def clone_chromosome(self):
        """Create a deep copy of the chromosome"""
        mp = deepcopy(self.main_program)
        adf_clones = [deepcopy(adf) for adf in self.adfs]
        return Chromosome(mp, adf_clones)

class CCGP:
    """CCGP Algorithm for task scheduling"""
    def __init__(self, pop_size, max_gen, task_list, vm_list, main_prog_len, num_adf, adf_len):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.task_list = task_list
        self.vm_list = vm_list
        self.main_prog_len = main_prog_len
        self.num_adf = num_adf
        self.adf_len = adf_len
        self.random = random.Random()
        
        # Populations for TSR and RSR
        self.pop_tsr = []
        self.pop_rsr = []
        
        # Fitness for each individual
        self.fitness_tsr = [0] * pop_size
        self.fitness_rsr = [0] * pop_size
        
        # Function and terminal sets
        self.function_set = ["+", "-", "*", "min", "max"]
        self.terminal_set = ["taskLen", "vmMips", "const1", "const2"]
        
        # Best individuals (representatives)
        self.rep_tsr = None
        self.rep_rsr = None
        
        # Best solution for printing/analysis
        self.best_tsr = None
        self.best_rsr = None
        self.best_fitness = float('inf')
        
        # Track the best assignment
        self.best_assignment = None
    
    def run(self):
        """Run the CCGP algorithm"""
        # Step 1: Initialization
        self.initialize_population(self.pop_tsr)
        self.initialize_population(self.pop_rsr)
        
        # Random representatives at initialization
        self.rep_tsr = self.pop_tsr[self.random.randint(0, self.pop_size-1)].clone_chromosome()
        self.rep_rsr = self.pop_rsr[self.random.randint(0, self.pop_size-1)].clone_chromosome()
        
        self.evaluate_populations(True)
        
        for gen in range(self.max_gen):
            # Steps 2~4: Evolution loop
            # Use the best individuals as representatives
            self.rep_tsr = self.pop_tsr[self.arg_min(self.fitness_tsr)].clone_chromosome()
            self.rep_rsr = self.pop_rsr[self.arg_min(self.fitness_rsr)].clone_chromosome()
            
            # Evolve both subpopulations
            self.evolve_population(self.pop_tsr, self.fitness_tsr, self.rep_rsr, True)
            self.evolve_population(self.pop_rsr, self.fitness_rsr, self.rep_tsr, False)
            
            # Update best global solution
            best_t = self.arg_min(self.fitness_tsr)
            best_r = self.arg_min(self.fitness_rsr)
            current_fitness = self.fitness(self.pop_tsr[best_t], self.pop_rsr[best_r])
            
            if current_fitness < self.best_fitness:
                self.best_tsr = self.pop_tsr[best_t].clone_chromosome()
                self.best_rsr = self.pop_rsr[best_r].clone_chromosome()
                self.best_fitness = current_fitness
                
                # Save the assignment that produces the best fitness
                self.best_assignment = self.schedule_by_heuristic(self.best_tsr)
    
    def initialize_population(self, pop):
        """Initialize a population of chromosomes"""
        pop.clear()
        for i in range(self.pop_size):
            main_prog = []
            for j in range(self.main_prog_len):
                if self.random.random() > 0.5:
                    main_prog.append(self.random.choice(self.function_set))
                else:
                    main_prog.append(self.random.choice(self.terminal_set))
            
            adfs = []
            for k in range(self.num_adf):
                adf = []
                for j in range(self.adf_len):
                    if self.random.random() > 0.5:
                        adf.append(self.random.choice(self.function_set))
                    else:
                        adf.append(self.random.choice(self.terminal_set))
                adfs.append(adf)
            
            pop.append(Chromosome(main_prog, adfs))
    
    def evaluate_populations(self, random_pair):
        """Evaluate both populations"""
        for i in range(self.pop_size):
            tsr = self.pop_tsr[i]
            rsr = self.pop_rsr[self.random.randint(0, self.pop_size-1)] if random_pair else self.rep_rsr
            self.fitness_tsr[i] = self.fitness(tsr, rsr)
        
        for i in range(self.pop_size):
            rsr = self.pop_rsr[i]
            tsr = self.pop_tsr[self.random.randint(0, self.pop_size-1)] if random_pair else self.rep_tsr
            self.fitness_rsr[i] = self.fitness(tsr, rsr)
    
    def evolve_population(self, pop, fit, other_rep, is_tsr):
        """Evolutionary step for a subpopulation"""
        # Step 5: Update building block frequencies for main program and ADFs
        freq = self.calculate_frequencies(pop)
        
        for i in range(self.pop_size):
            x = pop[i]
            # Step 6: Mutation (DE/rand-to-best/1 with frequency-based assignment)
            x_best = pop[self.arg_min(fit)]
            
            r1 = i
            while r1 == i:
                r1 = self.random.randint(0, self.pop_size-1)
            
            r2 = i
            while r2 == i or r2 == r1:
                r2 = self.random.randint(0, self.pop_size-1)
            
            xr1 = pop[r1]
            xr2 = pop[r2]
            
            mutant = self.frequency_based_mutation(x, x_best, xr1, xr2, freq)
            
            # Step 7: Crossover (binomial)
            trial = self.crossover(x, mutant)
            
            # Step 8: Selection
            trial_fitness = self.fitness(trial, other_rep) if is_tsr else self.fitness(other_rep, trial)
            if trial_fitness < fit[i]:
                pop[i] = trial
                fit[i] = trial_fitness
    
    def frequency_based_mutation(self, xi, x_best, xr1, xr2, freq):
        """Frequency-based mutation operation"""
        F = self.random.random()
        main_prog = []
        for j in range(self.main_prog_len):
            # Mutation probability
            mutate = self.random.random() < F
            if mutate:
                # Frequency-based assignment for main program
                main_prog.append(self.select_by_frequency(freq, True))
            else:
                main_prog.append(xi.main_program[j])
        
        adfs = []
        for k in range(self.num_adf):
            adf = []
            for j in range(self.adf_len):
                mutate = self.random.random() < F
                if mutate:
                    # Random assignment for ADFs
                    if self.random.random() > 0.5:
                        adf.append(self.random.choice(self.function_set))
                    else:
                        adf.append(self.random.choice(self.terminal_set))
                else:
                    adf.append(xi.adfs[k][j])
            adfs.append(adf)
        
        return Chromosome(main_prog, adfs)
    
    def crossover(self, xi, mutant):
        """Crossover operation (binomial)"""
        CR = self.random.random()
        child_main = []
        for j in range(self.main_prog_len):
            if self.random.random() < CR:
                child_main.append(mutant.main_program[j])
            else:
                child_main.append(xi.main_program[j])
        
        child_adfs = []
        for k in range(self.num_adf):
            child_adf = []
            for j in range(self.adf_len):
                if self.random.random() < CR:
                    child_adf.append(mutant.adfs[k][j])
                else:
                    child_adf.append(xi.adfs[k][j])
            child_adfs.append(child_adf)
        
        return Chromosome(child_main, child_adfs)
    
    def select_by_frequency(self, freq, main_program):
        """Selects a symbol by its frequency in the population"""
        # Only use main program frequencies for main program positions
        symbols = self.function_set + self.terminal_set
        
        total = sum(freq.get(s, 0) for s in symbols)
        r = self.random.randint(0, max(1, total) - 1)
        cum = 0
        for s in symbols:
            cum += freq.get(s, 0)
            if r < cum:
                return s
        
        # Fallback
        return self.random.choice(symbols)
    
    def calculate_frequencies(self, pop):
        """Calculates the frequency of each symbol in the population"""
        freq = {}
        for c in pop:
            for s in c.main_program:
                freq[s] = freq.get(s, 0) + 1
            for adf in c.adfs:
                for s in adf:
                    freq[s] = freq.get(s, 0) + 1
        return freq
    
    def arg_min(self, arr):
        """Returns the index of the minimum value in the array"""
        min_val = float('inf')
        idx = 0
        for i in range(len(arr)):
            if arr[i] < min_val:
                min_val = arr[i]
                idx = i
        return idx
    
    def fitness(self, tsr, rsr):
        """Fitness function as average SLR over all instances"""
        # For demonstration, use only one instance (task_list)
        # In full implementation, loop over multiple training instances if available.
        schedule = self.schedule_by_heuristic(tsr)
        return self.calculate_slr(schedule)
    
    def schedule_by_heuristic(self, chromosome):
        """Simulate scheduling by the evolved heuristic.
        This is a stub and should be replaced by actual evaluation of the expression trees.
        For now, assign tasks in round-robin.
        """
        # TODO: Replace with expression tree evaluation
        assignment = []
        for i in range(len(self.task_list)):
            assignment.append(i % len(self.vm_list))
        return assignment
    
    def calculate_slr(self, assignment):
        """Calculate the SLR (schedule length ratio) as defined in the algorithm"""
        # 1. Compute the makespan (schedule length)
        vm_processing = [0.0] * len(self.vm_list)
        for i in range(len(self.task_list)):
            c = self.task_list[i]
            vm_idx = assignment[i]
            exec_time = c.length / self.vm_list[vm_idx].mips
            vm_processing[vm_idx] += exec_time
        
        makespan = max(vm_processing)
        
        # 2. Compute minimum possible makespan (critical path with min execution time)
        # For cloud, assume CPMIN = all tasks (no dependencies) and min VM for each
        min_makespan = 0
        for c in self.task_list:
            min_exec = float('inf')
            for vm in self.vm_list:
                exec_time = c.length / vm.mips
                if exec_time < min_exec:
                    min_exec = exec_time
            min_makespan += min_exec
        
        # SLR = makespan / minMakespan
        return makespan / min_makespan
    
    def print_evaluation_metrics(self):
        """Print evaluation metrics for the best solution"""
        print(f"Best Solution SLR: {self.best_fitness}")
        
        if self.best_assignment is None:
            print("No best assignment found.")
            return {}
        
        # Calculate VM loads for the best solution
        vm_load = [0.0] * len(self.vm_list)
        vm_exec_times = [0.0] * len(self.vm_list)
        
        for i in range(len(self.task_list)):
            task = self.task_list[i]
            vm_index = self.best_assignment[i]
            vm = self.vm_list[vm_index]
            
            # Calculate execution time
            exec_time = task.length / vm.mips
            vm_load[vm_index] += task.length
            vm_exec_times[vm_index] += exec_time
        
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
            'slr': self.best_fitness,
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
        metrics_to_plot = ['makespan', 'throughput', 'slr', 'load_balance_variance']
        values = [metrics['makespan'], metrics['throughput'], metrics['slr'], metrics['load_balance_variance']]
        plt.bar(metrics_to_plot, values, color=['green', 'blue', 'purple', 'orange'])
        plt.title("Key Performance Metrics")
        plt.ylabel("Value")
        
        plt.tight_layout()
        plt.savefig("ccgp_results.png")
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
    
    # Initialize CCGP scheduler
    pop_size = 20
    max_gen = 50
    main_prog_len = 5
    num_adf = 2
    adf_len = 3
    
    print("Initializing CCGP scheduler...")
    start_time = time.time()
    
    ccgp = CCGP(pop_size, max_gen, tasks, vms, main_prog_len, num_adf, adf_len)
    
    # Run the scheduler
    print("Running CCGP optimization...")
    ccgp.run()
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    metrics = ccgp.print_evaluation_metrics()
    
    # Visualize results
    ccgp.visualize_results(metrics)
    
    # Output task assignments
    if ccgp.best_assignment:
        print("\nTask Assignments (Task ID -> VM ID):")
        for i, vm_id in enumerate(ccgp.best_assignment):
            if i < 10:  # Just show first 10 for brevity
                print(f"Task {tasks[i].task_id} -> VM {vms[vm_id].vm_id}")
        print("...")

if __name__ == "__main__":
    main()