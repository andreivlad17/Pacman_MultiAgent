# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math
import operator

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        foodList = newFood.asList()
        foodCount = len(foodList)
        closestDistance = float("inf")  # Initializeaza ca distanta maxima posibila

        if foodCount == 0:  # Daca nu este mancare, nu va exista reper pentru distanta, deci cea mai
            closestDistance = 0  # apropiata distanta este 0
        else:
            # Pentru fiecare punct de mancare, cea mai mica distanta va fi minimul dintre distanta minima curenta si
            # distanta Manhattan dintre pozitia curenta si punctul de mancare curent + foodCount * 100, unde
            # foodCount * 100 va influenta decizia prin cantitatea de mancare ramasa, pentru prioritizarea actiunilor
            for food in foodList:
                closestDistance = min(manhattanDistance(food, newPos) + foodCount * 100, closestDistance)

        # Actualizeaza scorul in functie de variabila decizionala din aceasta parte a metodei, closestDistance.
        # Cu cat scorul ramas este mai mare, cu atat actiunea analizata este mai recomandata
        score = -closestDistance

        # Verific daca exista fantome care sa aiba distanta pe grid mai micasau egala cu 1 sau
        # 2(prin teste am observat ca cresc sansele daca se alege random), caz in care scorul va fi scazut,
        # in asa fel incat sa nu se faca urmatoarea miscare, care ar duce la pierderea rundei
        for i in range(1, len(newGhostStates) + 1):
            ghostPosition = successorGameState.getGhostPosition(i)
            if manhattanDistance(newPos, ghostPosition) < random.randint(1, 2):
                score -= float("inf")
                break

        if action == 'Stop':
            return float("-inf")

        return score + random.randint(-6, 6)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    PACMAN_INDEX = 0

    def utility(self, gameState):
        """
        Functia de utilitate care calculeaza valoarea unei actiuni finale
        """
        return self.evaluationFunction(gameState)

    def terminalTest(self, gameState, depth):
        """
        Functia verifica daca starea actiunii actuale este una finala sau daca s-a atins adancimea(depth-ul)
        setat initial
        """
        return (len(gameState.getLegalActions(0)) == 0) or self.depth == depth

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1____

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action_____

        gameState.getNumAgents():
        Returns the total number of agents in the game _____

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        maxValue = -99999999
        bestAction = None  # cea mai buna actiune care poate fi aleasa

        for action in gameState.getLegalActions(self.PACMAN_INDEX):
            # se itereaza prin lista de actiuni pe care le poate face pacman si se cauta cea cu valoare MAXIMA
            # valoarea actiunilor este data de decizia adversarului, astfel se aplica functia minValue pe actiuni
            tempValue = self.minValue(1, gameState.generateSuccessor(0, action), 0)
            if tempValue > maxValue:
                maxValue = tempValue
                bestAction = action  # selectam actiunea cu valoarea cea mai mare(actiunea cea mai buna)

        return bestAction

    def maxValue(self, agentIndex, gameState, depth):  # functia pacman-ului
        if self.terminalTest(gameState, depth):  # se verifica daca actiunea este una finala
            return self.utility(gameState)
        v = -99999999
        for action in gameState.getLegalActions(agentIndex):
            # se itereaza prin lista de actiuni pe care le poate face pacman si se cauta cea cu valoare MAXIMA
            # valoarea actiunilor este data de decizia adversarului
            # intrucat adversarul este o fantoma(oponent), se apeleaza functia minValue
            successorState = gameState.generateSuccessor(agentIndex, action)  # starea succesorului
            tempValue = self.minValue(agentIndex + 1, successorState, depth)
            v = tempValue if tempValue > v else v  # valoarea maxima
        return v

    def minValue(self, agentIndex, gameState, depth):  # functia fantomelor
        if self.terminalTest(gameState, depth):  # se verifica daca actiunea este una finala
            return self.utility(gameState)
        v = 99999999
        if agentIndex < gameState.getNumAgents() - 1:
            # agentul curent este o fantoma diferita de ultima
            # intrucat agentul este o fantoma diferita de ultima, urmatorul agent explorat va fi tot o fantoma
            # se itereaza prin lista de actiuni pe care le poate face fantoma si se cauta cea cu valoare MINIMA
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)  # starea succesorului
                tempValue = self.minValue(agentIndex + 1, successorState, depth)
                v = tempValue if tempValue < v else v  # valoarea minima
        else:
            # ultima fantoma, urmeaza pacman
            # se itereaza prin lista de actiuni pe care le poate face pacman
            # intrucat urmeaza pacman se va aplica functia maxValue, insa se va selecta valoarea minima a actiunilor
            # intrucat agentul curent este o fantoma
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)  # starea succesorului
                tempValue = self.maxValue(self.PACMAN_INDEX, successorState,
                                          depth + 1)  # incrementare depth(adancime)
                v = tempValue if tempValue < v else v  # valoarea minima
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        v = self.maxValue(0, 0, state, float("-inf"), float("inf"))
        return v[0]

    def alphaBetaSearch(self, index, depth, state, alpha, beta):
        if state.isLose() or state.isWin() or depth is self.depth * state.getNumAgents():
            return self.evaluationFunction(state)
        if index == 0:
            return self.maxValue(index, depth, state, alpha, beta)[1]
        else:
            return self.minValue(index, depth, state, alpha, beta)[1]

    def maxValue(self, index, depth, state, alpha, beta):
        if state.isLose() or state.isWin() or depth is self.depth * state.getNumAgents():
            return self.evaluationFunction(state)
        value = ("max", float("-inf"))
        for action in state.getLegalActions(index):
            # value va lua valoarea tuplei cu valoare flotanta(index 1) maxima
            value = max(value, (action, self.alphaBetaSearch((depth + 1) % state.getNumAgents(), depth + 1,
                                                             state.generateSuccessor(index, action), alpha, beta)),
                        key=operator.itemgetter(1))

            # Pruning - nu functioneaza cu >= ca in indrumator
            if value[1] > beta:
                return value
            else:
                alpha = max(alpha, value[1])

        return value

    def minValue(self, index, depth, state, alpha, beta):
        if state.isLose() or state.isWin() or depth is self.depth * state.getNumAgents():
            return self.evaluationFunction(state)
        value = ("min", float("inf"))
        for action in state.getLegalActions(index):
            # value va lua valoarea tuplei cu valoare flotanta(index 1) minima
            value = min(value, (action, self.alphaBetaSearch((depth + 1) % state.getNumAgents(), depth + 1,
                                                             state.generateSuccessor(index, action), alpha, beta)),
                        key=operator.itemgetter(1))

            # Pruning - nu functioneaza cu <= ca in indrumator
            if value[1] < alpha:
                return value
            else:
                beta = min(beta, value[1])

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()

    foodDistance = 0

    returnedScore = currentGameState.getScore()

    for food in foodList:
        foodDistance += 2.5 * manhattanDistance(pacmanPosition, food)

    # Verifica daca exista fantome in apropiere. In cazul in care o fantoma este la o distanta mai mica sau egala
    # cu 4, variabila decizionala ghostProximity va fi diferita de 0, deci va influenta decizia mai mult sau mai putin,
    # in functie de distanta rezultata
    ghostProximity = 0
    for currentGhost in currentGameState.getGhostPositions():
        dist = max(4 - manhattanDistance(pacmanPosition, currentGhost), 0)
        ghostProximity += dist**3

    # Principalul obiectiv in acest joc este adunarea mancarii, de aceea agentul va trebui sa prioritizeze
    # punctele de mancare si in functie de cantitatea acestora la un anumit moment
    returnedScore -= 100 * len(foodList)
    if currentGameState.isLose():
        returnedScore -= 5000
    elif currentGameState.isWin():
        returnedScore += 5000

    returnedScore -= foodDistance
    returnedScore -= ghostProximity
    # In cazul in care nu exista mancare care sa duca la luarea unei decizii sau distanta prea mica fata de fantome,
    # Pacman ar ramane blocat intre 2 alegeri. Pentru a evita acest lucru a fost introdusa o valoare random (in limite
    # testate, (-6, 6) avand average score cu ~100-200 mai mare decat alte valori)
    # Aceasta valoare adaugata va duce la alegerea uneia dintre cele 2 stari
    returnedScore += random.randint(-6, 6)

    return returnedScore    # Scorul rezultat in urma analizei tuturor variabilelor decizionale


# Abbreviation
better = betterEvaluationFunction
