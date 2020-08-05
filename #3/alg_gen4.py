from pyeasyga import pyeasyga

data = ['x1', 'x2', 'x3', 'x4']

ga = pyeasyga.GeneticAlgorithm (data)

def fitness(individual, data):
    suma = 0
    x1 = individual[0]
    x2 = individual[1]
    x3 = individual[2]
    x4 = individual[3]
    if not x1 or x2 or x4:
        suma += 1
    if not x2 or x3 or x4:
        suma += 1
    if x1 or not x3 or x4:
        suma += 1
    if x1 or not x2 or not x4:
        suma += 1
    if x2 or not x3 or not x4:
        suma += 1
    if not x1 or x3 or not x4:
        suma += 1
    if x1 or x2 or x3:
        suma += 1
    return suma
ga.fitness_function = fitness
ga.run()
print(ga.best_individual())