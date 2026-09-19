import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    cobertura, redundancia, exposicion = problem.score_components(configuration)
    return cobertura - redundancia - exposicion


def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    actual = initial_configuration
    puntaje_actual = configuration_score(problem, actual)
    evaluaciones = 1
    historial = [actual]
    iteracion = 0

    while iteracion < max_iterations:
        vecinos = problem.neighbors(actual)
        mejor_vecino = None
        mejor_puntaje = puntaje_actual
        for vecino in vecinos:
            puntaje_vecino = configuration_score(problem, vecino)
            evaluaciones += 1
            if puntaje_vecino > mejor_puntaje:
                mejor_puntaje = puntaje_vecino
                mejor_vecino = vecino

        iteracion += 1
        if mejor_vecino is None:
            break

        actual = mejor_vecino
        puntaje_actual = mejor_puntaje
        historial.append(actual)

    return OptimizationResult(
        best_configuration=actual,
        best_score=puntaje_actual,
        evaluations=evaluaciones,
        iterations=iteracion,
        history=historial,
    )


def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    # TODO: Add your code here
    temperature = initial_temperature * (cooling_rate ** iteration)
    return temperature


def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9

    # TODO: Add your code here
    actual = initial_configuration
    iteracion = 1
    mejor = actual
    historial = [actual]
    temperature = initial_temperature
    while iteracion < max_iterations and temperature > minimum_temperature:
        candidato = rng.choice(problem.neighbors(actual))
        delta = configuration_score(problem, candidato) - configuration_score(problem, actual)
        if delta > 0:
            actual = candidato
            mejor = candidato
            historial.append(actual)
        else:
            acceptance_probability = math.exp(delta / temperature)
            if rng.random() < acceptance_probability:
                actual = candidato
                historial.append(actual)
        
        temperature = cooling_schedule(initial_temperature, cooling_rate, iteracion)
        iteracion += 1
    
    return OptimizationResult(
        best_configuration=mejor,
        best_score=configuration_score(problem, mejor),
        evaluations=iteracion*2, # Es 2 veces las iteraciones porque por cada iteracion se evalua en actual y el candidato
        iterations=iteracion,
        history=historial,
    )
                
            
    raise NotImplementedError("Punto 2: implemente simulated_annealing")


def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2

    corte = rng.randint(1, len(parent1) - 1)
    hijo1 = parent1[:corte] + parent2[corte:]
    hijo2 = parent2[:corte] + parent1[corte:]
    return hijo1, hijo2


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    if rng.random() >= mutation_probability:
        return individual

    activos = [indice for indice, bit in enumerate(individual) if bit]
    inactivos = [indice for indice, bit in enumerate(individual) if not bit]
    if not activos or not inactivos:
        return individual

    indice_activo = rng.choice(activos)
    indice_inactivo = rng.choice(inactivos)
    mutado = list(individual)
    mutado[indice_activo] = 0
    mutado[indice_inactivo] = 1
    return tuple(mutado)


def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    poblacion = problem.initial_population(population_size, rng)
    puntajes = [configuration_score(problem, individuo) for individuo in poblacion]
    evaluaciones = len(poblacion)

    mejor_indice = max(range(len(poblacion)), key=lambda indice: puntajes[indice])
    mejor_global = poblacion[mejor_indice]
    mejor_puntaje_global = puntajes[mejor_indice]

    historial = [mejor_global]
    iteracion = 0

    while iteracion < generations:
        orden = sorted(range(len(poblacion)), key=lambda indice: puntajes[indice], reverse=True)
        elite = [poblacion[indice] for indice in orden[:elite_size]]

        nueva_poblacion = list(elite)
        while len(nueva_poblacion) < population_size:
            padre1 = problem.tournament_select(poblacion, puntajes, rng)
            padre2 = problem.tournament_select(poblacion, puntajes, rng)
            hijo1, hijo2 = one_point_crossover(padre1, padre2, rng)

            hijo1 = problem.repair_configuration(hijo1, rng)
            hijo1 = swap_mutation(hijo1, mutation_probability, rng)
            nueva_poblacion.append(hijo1)

            if len(nueva_poblacion) < population_size:
                hijo2 = problem.repair_configuration(hijo2, rng)
                hijo2 = swap_mutation(hijo2, mutation_probability, rng)
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion
        puntajes = [configuration_score(problem, individuo) for individuo in poblacion]
        evaluaciones += len(poblacion)

        mejor_indice = max(range(len(poblacion)), key=lambda indice: puntajes[indice])
        if puntajes[mejor_indice] > mejor_puntaje_global:
            mejor_global = poblacion[mejor_indice]
            mejor_puntaje_global = puntajes[mejor_indice]

        historial.append(mejor_global)
        iteracion += 1

    return OptimizationResult(
        best_configuration=mejor_global,
        best_score=mejor_puntaje_global,
        evaluations=evaluaciones,
        iterations=iteracion,
        history=historial,
    )
