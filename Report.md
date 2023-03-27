[//]: # (Image References)

[image1]: imgs/plots.png "Plotted Scores"

# Udacity Project Two Continuous Control
The continuous control project is using Deep Deterministic Policy Gradient to train an agent to control the reacher, a robotic arm to follow the target in an unity environment.

## The Environment Description
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In order to solve the environment, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

## Learning Algorithm
I have used Deep Deterministic Policy Gradient (DDPG) algorithm to train the agent. This algorithm is based on an Actor-Critic architecture, where the actor will have a deterministic policy, that differs from PPO, and the critic will evaluate this policy. This is a off-policy, so it uses experience replay buffer to train the actor, and the critic is updated using the TD-error. Here was also used the fixed target for stabilizing the training, and the update happens after 20 iterations with the environment, to buffer this returns and the networks would be trained for 8 times (hyper parameter to avoid some instability and overfitting, training too much lead the agent to loose acumulated rewards). The number of episodes chosen was 1.000, but managed to solve with less than 200 episodes, and also was set the max_iter with the environment on 1.000 to avoid too long iteration that wont lead to any result.


### The Actor-Critic Network Architecture 

    Actor(
        Input size of 33 (states)
        Fully Connected layer: input_size = 33, output_size = 128 (Activation ReLU)
        Batch Norm 1D: 128
        Fully Connected layer: input_size = 128, output_size = 256 (Activation ReLU)
        Fully Connected layer: input_size = 256, output_size = 4 (Output continuous values for each joint)
    )

    Critic(
        Input size of 33 (states)
        Fully Connected layer: input_size = 33, output_size = 128 (Activation ReLU)
        Batch Norm 1D: 128
        Fully Connected layer: input_size = 128 + 4, output_size = 256 (Activation ReLU)
        Fully Connected layer: input_size = 256, output_size = 1 (Output the Q value)
    )

### hyper-parameter

    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.95            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-4         # learning rate of the actor 
    LR_CRITIC = 1e-3        # learning rate of the critic
    WEIGHT_DECAY = 0.       # L2 weight decay
    TRAIN_STEP = 20         # How long to wait until update network
    K_TRAIN = 8             # How many times to update the network (learn)

    The value of *GAMMA*, is responsible for the discounting the rewards for the episodes, so the nearest ones have more say in the final reward
    

## Train The Network
    Episode 50	Average Score: 4.96
    Episode 100	Average Score: 10.02
    Episode 150	Average Score: 23.55
    Environment solved in 176 episodes, mean score: 30.10

    Here was chosen the second version with 20 agents, and the final score was an average over all the agents in the last 100 episodes.

## Plotted Results

![Plotted Scores][image1]


## Ideas for Future Work
The Agent had a very nice performance, and was able to solve the environment in few steps, less than 200 !
Even thought, a few tweaks can be done, as change a little hyper parameters like the learning rate or the updates in the network, this number was critical on the convergence, les than 10 but more than 5. Also the noise parameters like the sigma and theta was critical for the convergence and any change would affect the exploration and thus the policy. I believe the most significant test would be also another method like D4PG or SAC (Soft Actor Critic).