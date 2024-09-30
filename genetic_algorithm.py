import random
import game
import copy
# In this case the heuristic will be in the form
# heuristic = {'score': score, 'height': height, 'holes': holes, 'bumpiness': bumpiness}
class GeneticAlgo:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, tournament_size, heuristic_keys, weight_range):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.heuristic_keys = heuristic_keys
        self.weight_range = weight_range
        self.env = game.TetrisEnv()
        
    def initialize_population(self):
        population = []
        for _ in range(population_size):
            chromosome = {}
            for key in heuristic_keys:
                chromosome[key] = random.uniform(weight_range[key][0], weight_range[key][1])
            population.append(chromosome)
        return population
    
    def evaluate_fitness(self, chromosome):
        total_score = 0
        num_simulations = 5
        for _ in range(num_simulations):
            score = self.run_tetris_bot(chromosome)
            total_score += score
        return total_score / num_simulations
            
    def run_tetris_bot(self, chromosome, limit=True):
        # Run simulation with chromosome
        self.env.reset()
        step = 0
        done = False
        while not done:
            step += 1
            # Get possible actions
            possible_actions = self.env.get_possible_actions()
            if not possible_actions:
                print("No possible actions.")
                done = True
                break
            # Choose action based on chromosome
            action = self.choose_action(chromosome, possible_actions)
            # Perform action
            done = self.env.step_heuristic(action['action_sequence'])
            if step > 250 and limit:
                break
        # Return score
        return self.env.score
            
    def choose_action(self, chromosome, possible_actions):
        for action in possible_actions:
            # Calculate heuristic values
            score = action['score'] * chromosome['score'] + action['height'] * chromosome['height'] + action['holes'] * chromosome['holes'] + action['bumpiness'] * chromosome['bumpiness']
            # Store score in action
            action['score'] = score
        # Choose action with highest score
        action = max(possible_actions, key=lambda x: x['score'])
        return action
   
    def tournament_selection(self, population, fitnesses):
        selected = []
        for _ in range(self.tournament_size):
            # Select random individual
            idx = random.randint(0, len(population) - 1)
            selected.append((population[idx], fitnesses[idx]))
        # Choose individual with highest fitness
        return max(selected, key=lambda x: x[1])[0]
    
    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        crossover_point = random.randint(1, len(self.heuristic_keys) - 1)
        child1 = {}
        child2 = {}
        for i, key in enumerate(self.heuristic_keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        return child1, child2
    
    def mutate(self, chromosome):
        for key in self.heuristic_keys:
            if random.random() < self.mutation_rate:
                mutation_strength = random.uniform(-.5, .5)
                chromosome[key] += mutation_strength
                # Ensure the weight is within the specified range
                chromosome[key] = max(self.weight_range[key][0], min(self.weight_range[key][1], chromosome[key]))
        return chromosome

    def run_genetic_algorithm(self):
        # Step 1: Initialize population
        population = self.initialize_population()
        
        for generation in range(self.generations):
            # Step 2: Evaluate fitness
            fitnesses = [self.evaluate_fitness(chromosome) for chromosome in population]
            
            # Print best fitness
            best_fitness = max(fitnesses)
            best_individual = population[fitnesses.index(best_fitness)]
            print(f"Generation {generation}: Best fitness: {best_fitness}")
            print(f"Best individual: {best_individual}")
            
            # Step 3: Create new population
            new_population = []
            
            # Step 3.5: Elitism
            new_population.append(best_individual)
            while len(new_population) < self.population_size:
                # Step 4: Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Step 5: Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Step 6: Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Replace old population with new population
            population = new_population
        
        # After all generations, return the best individual
        fitnesses = [self.evaluate_fitness(chromosome) for chromosome in population]
        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        print(f"Final best individual: {best_individual}")
        print(f"Final best fitness: {best_fitness}")
        return best_individual, best_fitness
            
        
        
    
population_size = 100
generations = 50
mutation_rate = 0.01
crossover_rate = 0.2
tournament_size = 5
heuristic_keys = ['score', 'height', 'holes', 'bumpiness']
weight_range = {
    'score': (-1, 1),
    'height': (-1, 1),
    'holes': (-1, 1),
    'bumpiness': (-1, 1)
}

# Initialize genetic algorithm
genetic_algo = GeneticAlgo(population_size, generations, mutation_rate, crossover_rate, tournament_size, heuristic_keys, weight_range)

# Run genetic algorithm
# best_individual, best_fitness = genetic_algo.run_genetic_algorithm()
heuristic = {'score': 0.612510657516274, 'height': -0.38949116713199194, 'holes': -0.8409069869764874, 'bumpiness': -0.09875057256130093}
while True:
    genetic_algo.run_tetris_bot(heuristic, limit=False)
