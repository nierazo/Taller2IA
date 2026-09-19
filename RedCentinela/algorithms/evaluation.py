from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    pendientes = state.pending_terminals
    distancia_objetivo = min(
        (state.layout.distance(state.defender_position, terminal) for terminal in pendientes),
        default=0,
    )
    distancia_intruso = state.layout.distance(state.defender_position, state.intruder_position)
    movilidad = len(state.get_legal_actions(0))

    valor = state.get_score()
    valor -= 25.0 * len(pendientes)
    if distancia_objetivo != float("inf"):
        valor -= 3.0 * distancia_objetivo
    if distancia_intruso != float("inf"):
        valor += min(distancia_intruso, 6.0)
        if distancia_intruso <= 1:
            valor -= 150.0
    valor += 2.0 * movilidad

    rta = max(-900.0, min(900.0, valor))
    return rta
