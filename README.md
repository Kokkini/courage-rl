# courage-rl
Using courage as an intrinsic reward to explore dangerous regions in the state-action space

## Motivation
Humans get better by tackling risky and challenging problems with courage. What will happen if we give courage to an AI agent? Let's find out

## Formulation
Definitions:
* Death: Terminating an episode before the maximum number of steps in reached with a non-positive reward for the last action. 
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

Looking at the average reward, we can see that PPO converged to the reward of 1 while courage could get the reward of 10. 

Looking at the max reward, we can see that PPO didn't even stumble upon the 10 reward, it didn't know that the 10 reward existed. Courage discovered the 1 reward but wasn't stuck at that reward because its love for danger pulled it away.

Below is the visualization of the danger level of all 5 actions at each state. On each square in the environment, there is a cross, with each hand of the cross 
corresponding to 4 actions up, down, left, right and the middle of the cross correspond to the action of staying still. Blue means low danger and yellow means high danger. 

<p align="center">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/16x16.jpg">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/16x16%20courage.gif">
</p>

At first, there is a wave of yellow, that's when the agent explored randomly and died a lot, leading to many state-action pairs being marked as dangerous. After a while of exploration, the agent learned the skill to avoid death so the yellow died down, only actions that lead to certain death remained bright. Now, if we 
look at where the new yellow pop up, we can see where the agent is exploring. It explored the whole maze and finally saw the 10 reward and converge as there is no danger left to distract it.

## Experiment 2: Generialization across environment parameters


<p align="center">
  <img width="480" src="https://github.com/Kokkini/courage-rl/blob/master/media/narrow_road_s.png">
</p>


<p align="center">
  Environment for training with wide road
</p>


<p align="center">
  <img width="480" src="https://github.com/Kokkini/courage-rl/blob/master/media/narrow_road_s_test.png">
</p>


<p align="center">
  Environment for evaluation with narrow road.
</p>

The goal is to drive the car to the green cylinder without touching the blue areas, which results in instant death.


Results:

* The baseline PPO agent does not generalize to the evaluation environment with narrow roads
* The courageous agent generalizes better to the evaluation environment because in training, it spent time driving really close to the blue areas without touching them.

|    | Courage | PPO |
| --- | --- | --- |
| training environment mean reward  | 4.33  | 4.33 |
| training environment success rate  | 1.00  | 1.00 |
| evaluation environment mean reward  | 2.73  | 1.65 |
| evaluation environment success rate  | 0.22  | 0.02 |


<p align="center">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/narrow_road_s_courage_train.gif">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/narrow_road_s_baseline_train.gif">
</p>


<p align="center">
  Training. Left: courage. Right: PPO
</p>



<p align="center">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/narrow_road_s_courage_worked.gif">
  <img width="240" src="https://github.com/Kokkini/courage-rl/blob/master/media/narrow_road_s_baseline_high.gif">
</p>


<p align="center">
  Evaluation. Left: courage. Right: PPO
</p>


## Experiment 3: Generialization across changes in dangerous zone positions

<p align="center">
  <img width="480" src="https://github.com/Kokkini/courage-rl/blob/master/media/car_in_box.png">
</p>


<p align="center">
  Environment for training with no obstacle blocking the way
</p>


<p align="center">
  <img width="480" src="https://github.com/Kokkini/courage-rl/blob/master/media/car_in_box_eval.png">
</p>


<p align="center">
  Environment for evaluation with a blocking obstacle
</p>

The goal is to drive the car to the green cylinder without touching the blue areas, which results in instant death.

This experiment is still in progress but what I expect to see is:
* The baseline PPO agent will not generalize
* The courageous agent will generalize to the evaluation environment because in training, it spent time driving really close to the blue areas without touching them.


## Experiment 4: Generialization across changes in goal positions

<p align="center">
  <img width="480" src="https://github.com/Kokkini/courage-rl/blob/master/media/road_and_field.png">
</p>


<p align="center">
  Environment for training with the goal randomly generated in the big yard below
</p>


<p align="center">
  <img width="480" src="https://github.com/Kokkini/courage-rl/blob/master/media/road_and_field_eval.png">
</p>


<p align="center">
  Environment for evaluation with the goal randomly generated in the narrow path above
</p>

The goal is to drive the car to the green cylinder without touching the blue areas, which results in instant death.

This experiment is still in progress but what I expect to see is:
* The baseline PPO agent will not generalize because it rarely set foot in the dangerous narrow path.
* The courageous agent will generalize to the evaluation environment because in training, it explored the dangerous narrow path well.



# Try it out
## Set up
```
git clone https://github.com/Kokkini/courage-rl.git
cd courage-rl
pip install -r requirements.txt
```

## Run an experiment with in the lava maze environment:
PPO + courage:
```
python train.py --config="experiments/courage.txt"
```
PPO without courage:
```
python train.py --config="experiments/baseline.txt" --baseline
```


## Run an experiment in Atari:
PPO + courage:
```
python train.py --config="experiments/general_experiment.txt" --visual_obs
```
PPO without courage:
```
python train.py --config="experiments/general_baseline_experiment.txt" --visual_obs --baseline
```

The default Atari environment is MontezumaRevenge-v0. You can change the environment by changing the "env" variable in the config file "experiments/general_experiment.txt"

## Result
The result of experiments are saved in ~/ray_results/PPO
