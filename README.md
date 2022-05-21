# An Application of Reinforcement Learning Techniques in Traditional Pathfinding  

## UCLA Masters of Applied Statistics Thesis  
**Author**: Britney Brown        
**Video of Training Results**: https://drive.google.com/file/d/1Egpwi1HS9AAGzSfrF6lGIGhYzd4Bsaqc/view?usp=sharing

## Summary of Results

### Pathfinding 
- Dijkstra solves each maze in 34 steps with 9.967 reward
- RL agent is successfully trained once it consistently receives a 9.967 reward

### Reinforcement Learning Agents
Trained three different agents (A2C, PPO, DQN) on each customizable maze: 
- A2C can reach the goal but is unable to converge on the optimal path ~ 65K training steps
- PPO generally converges after 50K training steps
- DQN consistently solves the maze and converges after ~35K training steps
	- w/ expert Dijkstra policy: solves/converges after ~2.5K training steps



