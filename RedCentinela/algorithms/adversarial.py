from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        self.nodes_evaluated = 0

        def valor(estado: GameState, agente: int, profundidad_restante: int) -> float:
            self.nodes_evaluated += 1
            if estado.is_win() or estado.is_lose() or profundidad_restante == 0:
                return evaluation_function(estado)

            siguiente_agente = (agente + 1) % estado.get_num_agents()
            acciones = estado.get_legal_actions(agente)
            if agente == 0:
                mejor_valor = float("-inf")
                for accion in acciones:
                    sucesor = estado.generate_successor(agente, accion)
                    valor_sucesor = valor(sucesor, siguiente_agente, profundidad_restante - 1)
                    if valor_sucesor > mejor_valor:
                        mejor_valor = valor_sucesor
                return mejor_valor
            else:
                peor_valor = float("inf")
                for accion in acciones:
                    sucesor = estado.generate_successor(agente, accion)
                    valor_sucesor = valor(sucesor, siguiente_agente, profundidad_restante - 1)
                    if valor_sucesor < peor_valor:
                        peor_valor = valor_sucesor
                return peor_valor

        self.nodes_evaluated += 1
        mejor_accion = None
        mejor_valor = float("-inf")
        for accion in state.get_legal_actions(0):
            sucesor = state.generate_successor(0, accion)
            valor_sucesor = valor(sucesor, 1, self.depth - 1)
            if valor_sucesor > mejor_valor:
                mejor_valor = valor_sucesor
                mejor_accion = accion
        return mejor_accion


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        self.nodes_evaluated = 0

        def valor(nodo, agente, restante, alpha, beta):
            self.nodes_evaluated += 1
            if nodo.is_win() or nodo.is_lose() or restante == 0:
                return evaluation_function(nodo)

            siguiente = (agente + 1) % nodo.get_num_agents()
            acciones = nodo.get_legal_actions(agente)

            if agente == 0:
                mejor = float("-inf")
                for accion in acciones:
                    sucesor = nodo.generate_successor(agente, accion)
                    mejor = max(mejor, valor(sucesor, siguiente, restante - 1, alpha, beta))
                    if mejor >= beta:
                        return mejor
                    alpha = max(alpha, mejor)
                return mejor

            mejor = float("inf")
            for accion in acciones:
                sucesor = nodo.generate_successor(agente, accion)
                mejor = min(mejor, valor(sucesor, siguiente, restante - 1, alpha, beta))
                if mejor <= alpha:
                    return mejor
                beta = min(beta, mejor)
            return mejor

        # la raíz también se evalúa
        self.nodes_evaluated += 1
        acciones_raiz = state.get_legal_actions(0)
        if not acciones_raiz:
            return None

        mejor_accion = acciones_raiz[0]
        mejor_valor = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for accion in acciones_raiz:
            sucesor = state.generate_successor(0, accion)
            valor_accion = valor(sucesor, 1, self.depth - 1, alpha, beta)
            if valor_accion > mejor_valor:
                mejor_valor = valor_accion
                mejor_accion = accion
            alpha = max(alpha, mejor_valor)

        return mejor_accion
