
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">z# Week 12 - Sequential Decision Making I
## Value and Policy Iteration Solutions</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Author: Massimo Caccia massimo.p.caccia@gmail.com <br>

The code was Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl <br>
and then from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">## 0. Preliminaries

Before we jump into the value and policy iteration excercies, we will test your comprehension of a Markov Decision Process (MDP). <br></div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### 0.1 Tic-Tac-Toe

Let's take a simple example: Tic-Tac-Toe (also known as Tic-tac-toe, noughts and crosses, or Xs and Os). Definition: it is a paper-and-pencil game for two players, X and O, who take turns marking the spaces in a 3×3 grid. The player who succeeds in placing three of their marks in a horizontal, vertical, or diagonal row is the winner.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://bjc.edc.org/bjc-r/img/3-lists/TTT1_img/Three%20States%20of%20TTT.png")
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**Question:** Imagine you were trying to build an agent for this game. Let's try to describe how we would model it. Specifically, what are the states, actions, transition function and rewards?</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**Answer:**<br>
The **state space** is a 3x3 Matrix or a vector of length 9 that indicates if a particular spot is: a) empty, b) taken by X or c) taken by O. <br>

The **actions** are on which of the 9 spot you can play (so there is 9 possible actions). Note that as the game evolves, some actions will become unavailable. <br>

An example of a **reward function** could return +1 if you win, -1 if you lose, and 0 for a draw.

The **transition function** is dictated by your opponent's strategy. <br></div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### 0.2 Recommender Systems

**Question:** In the last class we discussed recommender systems. Imagine that you would like to model the recommendation process overe time as an MDP. How would you do it?</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">**Answer:**

**States:** You would like the state to encode the user's preferences. There are different ways of doing so. Here is one: the state lists all the items previously consumed by the user.

**Actions:** which item to recommend (item 1, item 2, ... item n). Number of actions is the number of items.

**Reward:** +1 if the user consumes the recommeded item, -1 if not.

**Transition Probabilities:** that will depend on the user.</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">## 1. Value Iteration</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">The exercises will test your capacity to **complete the value iteration algorithm**.

You can find details about the algorithm at slide 46 of the [slide](http://www.cs.toronto.edu/~lcharlin/courses/80-629/slides_rl.pdf) deck. <br>

The algorithm will be tested on a simple Gridworld similar to the one presented at slide 12.</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### 1.1 Setup</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #imports

!wget -nc https://raw.githubusercontent.com/lcharlin/80-629/master/week12-MDPs/gridWorldGame.py
    
import numpy as np
from gridWorldGame import standard_grid, negative_grid, print_values, print_policy
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Let's set some variables. <br>
`SMALL_ENOUGH` is a threshold we will utilize to determine the convergence of value iteration<br>
`GAMMA` is the discount factor denoted $\gamma$ in the slides (see slide 36) <br>
`ALL_POSSIBLE_ACTIONS` are the actions you can take in the GridWold, as in slide 12. In this simple grid world, we will have four actions: Up, Down, Right, Left. <br>
`NOISE_PROB` defines how stochastic the environement is. It is the probability that the environment takes you where a random action would. </div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            SMALL_ENOUGH = 1e-3 # threshold to declare convergence
GAMMA = 0.9         # discount factor
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R') # Up, Down, Left, Right
NOISE_PROB = 0.1    # Probability of the agent not reaching it's intended goal after an action
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Now we will set up a the Gridworld. <br>
</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            grid = standard_grid(noise_prob=NOISE_PROB)
print("rewards:")
print_values(grid.rewards, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">There are three absorbing states: (0,3),(1,3), and (1,1)</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Next, we will define a random inital policy $\pi$. <br>
Remember that a policy maps states to actions $\pi : S \rightarrow A$.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Note that there is no policy in the absorbing/terminal states (hence the Not Available "N/A")</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Next, we will randomly initialize the value fonction</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.random.seed(1234) # make sure this is reproducable

V = {}
states = grid.all_states()
for s in states:
    # V[s] = 0
    if s in grid.actions:
        V[s] = np.random.random()
    else:
        # terminal state
        V[s] = 0

# initial value for all states in grid
print_values(V, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Note that we set to Null the values of the terminal states. <br> 
For the print_values() function to compile, we set them to 0.</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### 1.2 Value iteration algorithms - code completion

You will now have to complete the Value iteration algorithm. <br>
Remember that, for each iteration, each state s need to have to be update with the formula:

$$
V(s) = \underset{a}{max}\big\{ \sum_{s'}  p(s'|s,a)(r + \gamma*V(s') \big\}
$$
Note that in the current gridWorld, p(s'|s,a) is deterministic. <br>
Also, remember that in value iteration, the policy is implicit. <br> Thus, you don't need to update it at every iteration. <br>
Run the algorithm until convergence.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            iteration=0
while True:
    print("VI iteration %d: " % iteration)
    print_values(V, grid)
    print("\n\n")
  
    biggest_change = 0
    for s in states:
        old_v = V[s]

        # V(s) only has value if it's not a terminal state
        if s in policy:
            new_v = float('-inf')

            # for each action
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                sprime = grid.current_state()
                #  - compute this V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
                v = r + GAMMA * V[sprime]
                if v > new_v: # is this the best action so far
                    new_v = v
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    print('\t biggest change is: %f \n\n' % biggest_change)
    if biggest_change < SMALL_ENOUGH:
        break
    iteration+=1
print_values(V, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Now that the value function is optimized, use it to find the optimal policy.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            deterministic_grid = standard_grid(noise_prob=0.)

for s in policy.keys():
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in ALL_POSSIBLE_ACTIONS:
        deterministic_grid.set_state(s)
        r = deterministic_grid.move(a)
        v = r + GAMMA * V[deterministic_grid.current_state()]
        if v > best_value:
            best_value = v
            best_a = a
    policy[s] = best_a
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Now print your policy and make sure it leads to the upper-right corner which is the termnial state returning the most rewards.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            print("values:")
print_values(V, grid)
print("\npolicy:")
print_policy(policy, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">## 2. Policy Iteration</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">You will be tested on your capacity to **complete the poliy iteration algorithm**. <br>
You can find details about the algorithm at slide 47 of the slide deck. <br>
The algorithm will be tested on a simple Gridworld similar to the one presented at slide 12. <br>
This Gridworld is however simpler because the MDP is deterministic. <br></div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">First we will define a random inital policy. <br>
Remember that a policy maps states to actions.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Next, we will randomly initialize the value fonction</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            np.random.seed(1234)

# initialize V(s) - value function
V = {}
states = grid.all_states()
for s in states:
    if s in grid.actions:
        V[s] = np.random.random()
    else:
        # terminal state
        V[s] = 0

# initial value for all states in grid
print_values(V, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Note that we set to Null the values of the terminal states. <br> 
For the print_values() function to compile, we set them to 0.</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### 2.2 Policy iteration - code completion

You will now have to complete the Policy iteration algorithm. <br>
Remember that the algorithm works in two phases. <br>
First, in the *policy evaluation* phase, the value function is update with the formula:

$$
V^\pi(s) =  \sum_{s'}  p(s'|s,\pi(s))(r + \gamma*V^\pi(s') 
$$
This part of the algorithm is already coded for you. <br>

Second, in the *policy improvement* step, the policy is updated with the formula:

$$
\pi'(s) = \underset{a}{arg max}\big\{ \sum_{s'}  p(s'|s,a)(r + \gamma*V^\pi(s') \big\}
$$

This is the part of code you will have to complete. <br>

Note that in the current gridWorld, p(s'|s,a) is deterministic. <br>
Run the algorithm until convergence.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            iteration=0
# repeat until the policy does not change
while True:
    print("values (iteration %d)" % iteration)
    print_values(V, grid)
    print("policy (iteration %d)" % iteration)
    print_policy(policy, grid)
    print('\n\n')

    # 1. policy evaluation step
    # this implementation does multiple policy-evaluation steps
    # this is different than in the algorithm from the slides 
    # which does a single one.
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]

            # V(s) only has value if it's not a terminal state
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a) # reward
                sprime = grid.current_state() # s' 
                V[s] = r + GAMMA * V[sprime]
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        if biggest_change < SMALL_ENOUGH:
            break

    #2. policy improvement step
    is_policy_converged = True
    for s in states:
        if s in policy:
            old_a = policy[s]
            new_a = None
            best_value = float('-inf')
            # loop through all possible actions to find the best current action
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                sprime = grid.current_state() 
                v = r + GAMMA * V[sprime]
                if v > best_value:
                    best_value = v
                    new_a = a
            if new_a is None: 
                print('problem')
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

    if is_policy_converged:
        break
    iteration+=1

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">Now print your policy and make sure it leads to the upper-right corner which is the termnial state returning the most rewards.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;"># </div>


    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    