# Este codigo fue recuperado de https://www.pypro.mx/
import random
from functools import partial
from deap import base, creator, tools

# Función para generar la lista de ubicaciones
def generate_locations(num_locations):
    locations = []
    for i in range(num_locations):
        locations.append((random.uniform(0, 1), random.uniform(0, 1)))  # Generate random location
    return locations

# Función para evaluar la distancia total de una ruta
def evaluate_route(individual, locations):
    distance = 0.0
    for i in range(len(individual)-1):
        x1, y1 = locations[individual[i]]
        x2, y2 = locations[individual[i+1]]
        distance += ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5  # Euclidean distance
    return distance,

# Definir tipos de fitness y cromosomas
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Configuración de la Toolbox de DEAP
toolbox = base.Toolbox()
toolbox.register("locations", generate_locations, num_locations=20)
toolbox.register("individual", tools.initPermutation, creator.Individual, len=20)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_route, locations=toolbox.locations())
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

# Configuración de los parámetros del Algoritmo Genético
population = toolbox.population(n=100)
cxpb, mutpb, ngen = 0.5, 0.2, 50

# Ejecutar el Algoritmo Genético
for gen in range(ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# Obtener la mejor solución al problema de la ruta del repartidor
best_solution = tools.selBest(population, k=1)[0]
best_distance = evaluate_route(best_solution, locations=generate_locations(20))[0]
print("La mejor ruta encontrada es {} con una distancia total de {}".format(best_solution, best_distance))