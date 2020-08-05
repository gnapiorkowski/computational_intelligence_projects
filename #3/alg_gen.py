from pyeasyga import pyeasyga
from matplotlib import pyplot as plt

data = [{'name': 'zegar', 'value': 100, 'weight': 7},
        {'name': 'obraz-pejzaz', 'value': 300, 'weight': 7},
        {'name': 'obraz-portret', 'value': 200, 'weight': 6},
        {'name': 'radio', 'value': 40, 'weight': 2},
        {'name': 'laptop', 'value': 500, 'weight': 5},
        {'name': 'lampka nocna', 'value': 70, 'weight': 6},
        {'name': 'srebrne sztucce', 'value': 100, 'weight': 1},
        {'name': 'porcelana', 'value': 250, 'weight': 3},
        {'name': 'figura z brazu', 'value': 300, 'weight': 10},
        {'name': 'skorzana torebka', 'value': 280, 'weight': 3},
        {'name': 'odkurzacz', 'value': 300, 'weight': 15}]
generacje = 100
ga = pyeasyga.GeneticAlgorithm(data,
                               population_size=200,
                               generations=generacje,
                               crossover_probability=0.8,
                               mutation_probability=0.05,
                               elitism=True,
                               maximise_fitness=True)


def fitness(individual, data):
    value_sum, weight_sum = 0, 0
    for selected, box in zip(individual, data):
        if selected:
            value_sum += box.get('value')
            weight_sum += box.get('weight')
    if weight_sum > 25:
        value_sum = 0
    return value_sum

ga.fitness_function = fitness

best_osobniki = []
fitness_means =[]

ga.create_initial_population()
ga.calculate_population_fitness()
ga.rank_population()

fitnesses =[]
for osobnik in ga.last_generation():
    fitnesses.append(osobnik[0])
fitness_means.append(sum(fitnesses)/len(fitnesses))
best_osobniki.append(ga.best_individual()[0])

#print('Generation 1: ', ga.best_individual())


ga.create_first_generation()
ga.calculate_population_fitness()
ga.rank_population()

fitnesses =[]
for osobnik in ga.last_generation():
    fitnesses.append(osobnik[0])
fitness_means.append(sum(fitnesses)/len(fitnesses))
best_osobniki.append(ga.best_individual()[0])

#print('Generation 2: ', ga.best_individual())


for i in range(generacje - 2):
    ga.create_next_generation()
    ga.calculate_population_fitness()
    ga.rank_population()

    fitnesses =[]
    for osobnik in ga.last_generation():
        fitnesses.append(osobnik[0])
    fitness_means.append(sum(fitnesses)/len(fitnesses))
    best_osobniki.append(ga.best_individual()[0])

    #print('Generation '+str(i+3)+': ', ga.best_individual())

print(fitness_means, best_osobniki)

plt.plot( range(generacje), fitness_means,'r', range(generacje), best_osobniki, 'b')
plt.show()