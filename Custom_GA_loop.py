# Implement the main loop of the evolutionary algorithm:
for gen in range(generations):
    # Select the next generation individuals:
    offspring = toolbox.select(population, len(population))

    # Clone the selected individuals:
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring:
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_p:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_p:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness:
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    args = [(ind, cor_matrix, Log_Likelihood, BIC, AIC) for ind in invalid_ind]
    fitnesses = toolbox.map(toolbox.evaluate, args)

    # Update the fitness values of the individuals:
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace population with the offspring:
    population[:] = offspring

    # Update the hall of fame with the generated individuals:
    hof.update(population)

    # Gather all the fitnesses in one list and print the stats:
    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    stats.record(gen=gen, evals=len(invalid_ind), std=std, min=min(fits), avg=mean, max=max(fits))

    print(stats.compile(population))

# print best solution found:
best = hof.items[0]  

print("-- Best Ever Individual = ", best[0])  
print("-- Best Ever items = ", best.items)
print("-- Best Ever Fitness = ", best.fitness.values)