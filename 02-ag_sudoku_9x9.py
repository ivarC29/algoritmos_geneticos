import numpy as np
import random

class SudokuGA:
    def __init__(self, sudoku, mutation_rate, n_individuals, n_selection, n_generations, verbose=True):
        self.sudoku = np.array(sudoku)
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.n_selection = n_selection
        self.n_generations = n_generations
        self.verbose = verbose
        self.target_size = 9 * 9

    def create_individual(self):
        individual = np.copy(self.sudoku)
        for i in range(9):
            for j in range(9):
                if individual[i][j] == 0:
                    individual[i][j] = random.randint(1, 9)
        return individual

    def create_population(self):
        return [self.create_individual() for _ in range(self.n_individuals)]

    def fitness(self, individual):
        fitness = 0
        for i in range(9):
            fitness += (9 - len(np.unique(individual[i])))  # Filas
            fitness += (9 - len(np.unique(individual[:, i])))  # Columnas
        for i in range(3):
            for j in range(3):
                fitness += (9 - len(np.unique(individual[i*3:(i+1)*3, j*3:(j+1)*3])))  # Subcuadrículas
        return fitness

    def selection(self, population):
        scores = [(self.fitness(ind), ind) for ind in population]
        scores.sort(key=lambda x: x[0])
        selected = [ind for _, ind in scores[:self.n_selection]]
        return selected

    def crossover(self, parent1, parent2):
        child = np.copy(parent1)
        for subgrid in range(9):
            if random.random() > 0.5:
                row_start = (subgrid // 3) * 3
                col_start = (subgrid % 3) * 3
                child[row_start:row_start+3, col_start:col_start+3] = parent2[row_start:row_start+3, col_start:col_start+3]
        return child

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            row, col = random.randint(0, 8), random.randint(0, 8)
            while self.sudoku[row][col] != 0:
                row, col = random.randint(0, 8), random.randint(0, 8)
            individual[row][col] = random.randint(1, 9)
        return individual

    def run_geneticalgo(self):
        population = self.create_population()
        for gen in range(self.n_generations):
            selected = self.selection(population)
            new_population = []

            for _ in range(self.n_individuals):
                parent1, parent2 = random.sample(selected, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                new_population.append(child)

            population = new_population

            best_individual = min(population, key=self.fitness)
            if self.verbose:
                print(f"Generación {gen + 1}: Mejor individuo (Errores: {self.fitness(best_individual)})")
                if self.fitness(best_individual) == 0:
                    break

        best_individual = min(population, key=self.fitness)
        print("\nMejor solución encontrada:")
        print(f"Aptitud (Errores): {self.fitness(best_individual)}")
        print(best_individual)

def main():
    sudoku = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    model = SudokuGA(
        sudoku=sudoku,
        mutation_rate=0.05,
        n_individuals=500,
        n_selection=50,
        n_generations=10000,
        verbose=True
    )
    model.run_geneticalgo()

if __name__ == "__main__":
    main()