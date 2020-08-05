from pyeasyga import pyeasyga
import random


generacje = 100


layout = ["############",
        "#S  #   #  #",
        "###   # ## #",
        "#   # #    #",
        "# # ##  ## #",
        "#  ##   #  #",
        "#     #   ##",
        "# #  ## #  #",
        "# ###   ## #",
        "# # ## # # #",
        "# #       E#",
        "############"]

MAX_STEPS = 40

for i in range(len(layout)):
    if layout[i].find('S') != -1:
        posy = i
        posx = layout[i].find('S')

data = {'layout':layout, 'max_steps':MAX_STEPS, 'posx':posx, 'posy':posy}

def create_individual(data):
    individual = []
    max_steps = data['max_steps']
    posx = data['posx']
    posy = data['posy']
    for i in range(max_steps):
        individual.append(random.randint(0, 3))
        while True:
            if individual[i] == 0:
                #UP and 
                if posy > 1:
                    posy -= 1
                    if layout[posy][posx] != '#': posy += 1
                    else: break
                else: individual[i] = random.randint(0,3)
            elif individual[i] == 1:
                #RIGHT
                if posx <10:
                    posx += 1
                    if layout[posy][posx] != '#': posx -= 1
                    else: break
                else: individual[i] = random.randint(0,3)
            elif individual[i] == 2:
                #DOWN
                if posy < 10:
                    posy += 1
                    if layout[posy][posx] != '#': posy -= 1
                    else: break
                else: individual[i] = random.randint(0,3)
            elif individual[i] == 3:
                #LEFT
                if posx > 1:
                    posx -= 1
                    if layout[posy][posx] != '#': posx += 1
                    else: break
                else: individual[i] = random.randint(0,3)

            individual[i] = random.randint(0,3)
    return individual

create_individual(data)

def mutate(individual):
    mutate_index = random.randrange(len(individual))
    individual[mutate_index] = (random.randint(0, 3))

def fitness(individual, data):
    start = False
    steps = 0
    posx = data["posx"]
    posy = data["posy"]
    layout = data["layout"]
    penalty = 0
    for step in individual:
        steps += 1
        if step == 0:
            #UP
            posy -= 1
        elif step == 1:
            #RIGHT
            posx += 1
        elif step == 2:
            #DOWN
            posy += 1
        elif step == 3:
            #LEFT
            posx -= 1
        if layout[posy][posx] == '#': return posx+posy+steps - penalty
        if layout[posy][posx] == 'E': return (data['max_steps'] - steps)*100
        #print(posx, posy, '|', layout[posy][posx], layout[posy][posx])
    return posx+posy+steps - penalty

#example = [1, 1, 2, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 2, 1, 2, 3, 1, 3, 3, 0, 2, 2, 3, 0]
#print('fitness: ', fitness(example, data))

ga = pyeasyga.GeneticAlgorithm(data,
                            population_size=500,
                            generations=generacje,
                            crossover_probability=0.8,
                            mutation_probability=0.9,
                            elitism=True,
                            maximise_fitness=True)

ga.fitness_function = fitness
ga.create_individual = create_individual
ga.mutate_function = mutate
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

print('Generation 1: ', ga.best_individual())


ga.create_first_generation()
ga.calculate_population_fitness()
ga.rank_population()

fitnesses =[]
for osobnik in ga.last_generation():
    fitnesses.append(osobnik[0])
fitness_means.append(sum(fitnesses)/len(fitnesses))
best_osobniki.append(ga.best_individual()[0])

print('Generation 2: ', ga.best_individual())


for i in range(generacje - 2):
    ga.create_next_generation()
    ga.calculate_population_fitness()
    ga.rank_population()

    fitnesses =[]
    for osobnik in ga.last_generation():
        fitnesses.append(osobnik[0])
    fitness_means.append(sum(fitnesses)/len(fitnesses))
    best_osobniki.append(ga.best_individual()[0])

    print('Generation '+str(i+3)+': ', ga.best_individual())
def stepsfit(individual, data):
    start = False
    steps = 0
    posx = data["posx"]
    posy = data["posy"]
    layout = data["layout"]
    penalty = 0
    for step in individual:
        steps += 1
        if step == 0:
            #UP
            posy -= 1
        elif step == 1:
            #RIGHT
            posx += 1
        elif step == 2:
            #DOWN
            posy += 1
        elif step == 3:
            #LEFT
            posx -= 1
        if layout[posy][posx] == '#': penalty += 1
        if layout[posy][posx] == 'E': penalty -= 1000
        #print(posx, posy, '|', layout[posy][posx], layout[posy][posx])
    return steps
    print('steps: ', stepsfit(ga.best_individual()[1], data))