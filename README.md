# courage-rl
Using courage as an intrinsic reward to explore dangerous regions in the state-action space

## Motivation
Humans get better by tackling risky and challenging problems with courage. What will happen if we give courage to an AI agent? Let's find out

## Formulation
Definitions:
* Death: Terminating and episode before the maximum number of steps in reached with a non-positive reward for the last action. 
This definition can be changed based on the environment.
* Danger: the danger of a state-action pair is the average discounted death rate of trajectories containing that state-action pair.
* Courage reward: if the next state is not death, courage reward is the danger of the state-action pair. If the next state is death, courage reward is death reward (a non-positive hyperparamter)

With this formulation, a courageous agent will want to perform dangerous actions while avoiding actions leading to certain death (courageous but not foolish).
When the agent perform dangerous actions for a while, its skills will get better, making the actions less and less dangerous. It will move on to other more dangerous actions
and improve its skills there. Dangerous regions are the regions that needs the most skills and courage incentivises the agent to focus experience collections in those regions. This leads to better and more robust skills.

A courageous agent optimize for the following reward:

r = c1 * r_env + c2 * r_courage

where: 
* r_env is the reward coming from the environment
* r_courange is the courage reward
* c1 and c2 are coefficients.

Courage is different from curiosity in that curiosity focuses on exploring less seen and less understood regions while courage focuses on exploring dangerous regions. They are not exclusive and both can be used at the same time on the same agent.

## Experiment 1: the path to success is paved with failure.

<p align="center">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/16x16.jpg">
</p>

In this experiment, the upper part of the environment is a maze filled with lava holes (red), leading to a big reward of 10 (yellow), 
the lower part is empty and safe but leading to a small reward of 1 (green). The agent (blue) starts at the upper right corner, with 5 actions
at every state: up, down, left, right, stay still. Walking into the lava holes will give a reward of 0 and reset the environment. Walking
into one of the reward will give the correspoding reward and reset the environment. The optimal strategy is going through the lava maze to get
the yellow reward. 

This poses significant challenges for usual RL methods such as PPO because: If the agent chooses the dangerous path, it will encounter countless deaths and
very few big reward, making the average reward of going in that path very low compared to the safe path to the small reward. PPO agents will choose the safe
path every time.

<p align="center">
  <img width="400" src="https://github.com/Kokkini/courage-rl/blob/master/media/mean%20reward%2016x16%20baseline%20vs%20courage.png">
  <img width="400" src="https://github.com/Kokkini/courage-rl/blob/master/media/max%20reward%2016x16%20baseline%20vs%20courage.png">
</p>

Looking at the average reward, we can see that PPO converged to the reward of 1 while courage could get the reward of 10. Looking at the max reward, we can see
that PPO didn't even stumble upon the 10 reward, it didn't know that the 10 reward existed.
