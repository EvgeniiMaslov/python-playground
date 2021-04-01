![](https://miro.medium.com/max/650/1*-XKVI5SAEpffNR7BusdvNQ.png)

## Content

1. [What is AI](#What-is-AI)

2. [Problem Solving Agents](#Problem-solving-Agents)

   2.1. [Well-defined problems and solutions](#Well-defined-problems-and-solutions)

   2.2. [Infrastructure for search algorithms](#Infrastructure for search algorithms)

   





### What is AI

Some definitions of artificial intelligence, organized into four categories:

| **Thinking Humanly**:                                        | Thinking Rationally:                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| "The exciting new effort to make computers think ... *machines with minds*, in the full and literal sense" (Haugeland, 1985)<br />“[The automation of] activities that we associate with human thinking, activities such as decision-making, problem solving, learning ...” (Bellman, 1978) | “The study of mental faculties through the use of computational models.” (Charniak and McDermott, 1985) <br />“The study of the computations that make it possible to perceive, reason, and act.” (Winston, 1992) |
| **Acting Humanly:**                                          | **Acting Rationally:**                                       |
| “The art of creating machines that perform functions that require intelligence when performed by people.” (Kurzweil, 1990)<br />“The study of how to make computers do things at which, at the moment, people are better.” (Rich and Knight, 1991) | “Computational Intelligence is the study of the design of intelligent agents.” (Poole et al., 1998)<br />“AI . . . is concerned with intelligent behavior in artifacts.” (Nilsson, 1998) |

**Acting Humanly: Turing Test**: 

The Turing Test, proposed by Alan Turing (1950), was designed to provide a satisfactory operational definition of intelligence. A computer passes the test if a human interrogator, after posing some written questions, cannot tell whether the written responses come from a person or from a computer. The computer would need to possess the following capabilities:

* **natural language processing** to enable it to communicate successfully in English;
* **knowledge representation** to store what it knows or hears;
* **automated reasoning** to use the stored information to answer questions and to draw new conclusions;
* **machine learning** to adapt to new circumstances and to detect and extrapolate patterns.

The so-called total Turing Test includes a video signal so that the interrogator can test the subject’s perceptual abilities, as well as the opportunity for the interrogator to pass physical objects “through the hatch.” To pass the total Turing Test, the computer will need:

* **computer vision** to perceive objects, and
* **robotics** to manipulate objects and move about.

These six disciplines compose most of AI.

**Rational agent definition: ** For each possible percept sequence, a rational agent should select an action that is expected to maximize its performance measure, given the evidence provided by the percept sequence and whatever built-in knowledge the agent has.



**Whirlwind tour of AI**:

* An agent is something that perceives and acts in an environment. The **agent function** for an agent specifies the action taken by the agent in response to any percept sequence. 
* The **performance measure** evaluates the behavior of the agent in an environment. A **rational agent** acts so as to maximize the expected value of the performance measure, given the percept sequence it has seen so far. 
* A **task environment** specification includes the performance measure, the external environment, the actuators, and the sensors. In designing an agent, the first step must always be to specify the task environment as fully as possible. 
* Task environments vary along several significant dimensions. They can be fully or partially observable, single-agent or multiagent, deterministic or stochastic, episodic or sequential, static or dynamic, discrete or continuous, and known or unknown. 
* The **agent program** implements the agent function. There exists a variety of basic agent-program designs reflecting the kind of information made explicit and used in the decision process. The designs vary in efficiency, compactness, and flexibility. The appropriate design of the agent program depends on the nature of the environment. 
* **Simple reflex agents** respond directly to percepts, whereas **model-based reflex agents** maintain internal state to track aspects of the world that are not evident in the current percept. **Goal-based agents** act to achieve their goals, and **utility-based agents** try to maximize their own expected “happiness.” 
* All agents can improve their performance through **learning**.





### Problem-solving Agents



Problem-solving agents use atomic representations, that is, states of the world are considered as wholes, with no internal structure visible to the problem-solving algorithms.

**Goal formulation**, based on the current situation and the agent’s performance measure, is the first step in problem solving. We will consider a goal to be a set of world states—exactly those states in which the goal is satisfied. The agent’s task is to find out how to act, now and in the future, so that it reaches a goal state.

**Problem formulation** is the process of deciding what actions and states to consider, given a goal.

The process of looking for a sequence of actions that reaches the goal is called **search**. A search algorithm takes a problem as input and returns a **solution** in the form of an action sequence. Once a solution is found, the actions it recommends can be carried out. This is called the **execution** phase. 

Thus, we have a simple “formulate, search, execute” design for the agent.



```pseudocode
function Simple-Problem-Solving-Agent(percept) return an action:
	persistent: seq, an action sequence, initially empty
				state, some description of current world state
				goal, a goal, initially null,
				problem, a problem formulation
	
	state <- UpdateState(state, percept)
	if seq is empty then:
		goal <- FormulateGoal(state)
		problem <- FormulateProblem(state, goal)
		seq <- Search(problem)
		if seq == failure then return a null action
	action <- First(seq)
	seq <- Rest(seq)
	return action
```





#### Well-defined problems and solutions



A problem can be defined formally by five components:

* The **initial state** that the agent starts in.

* A description of the possible **actions** available to the agent. Given a particular state s, ACTIONS(s) returns the set of actions that can be executed in s. We say that each of these actions is applicable in s.

* A description of what each action does; the formal name for this is the **transition model**, specified by a function RESULT(s, a) that returns the state that results from doing action a in state s. We also use the term **successor** to refer to any state reachable from a given state by a single action.

  Together, the initial state, actions, and transition model implicitly define the **state space** of the problem—the set of all states reachable from the initial state by any sequence of actions.

* The **goal test**, which determines whether a given state is a goal state. Sometimes there is an explicit set of possible goal states, and the test simply checks whether the given state is one of them.

* A **path cost** function that assigns a numeric cost to each path. The problem-solving agent chooses a cost function that reflects its own performance measure.



Having formulated some problems, we now need to solve them. A solution is an action sequence, so search algorithms work by considering various possible action sequences. The possible action sequences starting at the initial state form a search tree with the initial state at the root; the branches are actions and the nodes correspond to states in the state space of the problem.

As the saying goes, algorithms that forget their history are doomed to repeat it. The way to avoid exploring redundant paths is to remember where one has been. To do this, we augment the TREE-SEARCH algorithm with a data structure called the explored set (also known as the closed list), which remembers every expanded node. Newly generated nodes that match previously generated nodes—ones in the explored set or the frontier—can be discarded instead of being added to the frontier. This algorithm is GRAPH-SEARCH.  

```pseudocode
function TREE-SEARCH(problem) returns a solution, or failure

	initialize the frontier using the initial state of problem
	loop do
        if the frontier is empty then return failure
        choose a leaf node and remove it from the frontier
        if the node contains a goal state then return the corresponding solution
        expand the chosen node, adding the resulting nodes to the frontier
        
        
function GRAPH-SEARCH(problem) returns a solution, or failure
    initialize the frontier using the initial state of problem
    initialize the explored set to be empty
    loop do
        if the frontier is empty then return failure
        choose a leaf node and remove it from the frontier
        if the node contains a goal state then return the corresponding solution
        add the node to the explored set
        expand the chosen node, adding the resulting nodes to the frontier
        	only if not in the frontier or explored set
```



#### Infrastructure for search algorithms

Search algorithms require a data structure to keep track of the search tree that is being constructed. For each node n of the tree, we have a structure that contains four components:

* n.STATE: the state in the state space to which the node corresponds; 
* n.PARENT: the node in the search tree that generated this node; 
* n.ACTION: the action that was applied to the parent to generate the node; 
* n.PATH-COST: the cost, traditionally denoted by g(n), of the path from the initial state to the node, as indicated by the parent pointers.

Given the components for a parent node, it is easy to see how to compute the necessary components for a child node. The function CHILD-NODE takes a parent node and an action and returns the resulting child node:

```pseudocode
function CHILD-NODE(problem, parent, action) returns a node
    return a node with
        STATE = problem.RESULT(parent.STATE, action),
        PARENT = parent, ACTION = action,
        PATH-COST = parent.PATH-COST + problem.STEP-COST(parent.STATE, action)
```

Now that we have nodes, we need somewhere to put them. The frontier needs to be stored in such a way that the search algorithm can easily choose the next node to expand according to its preferred strategy. The appropriate data structure for this is a queue. The operations on a queue are as follows:

* EMPTY?(queue) returns true only if there are no more elements in the queue. 
* POP(queue) removes the first element of the queue and returns it. 
* INSERT(element, queue) inserts an element and returns the resulting queue.