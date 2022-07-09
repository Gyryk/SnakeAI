# SnakeAI
## The classic game of snake, but it is controlled by a Deep RL algorithm

The algorithm is currently only learning for ~500 runs due to a hidden layer constraint and an error in the algorithm that causes the snake to loop and collide into itself. To solve that, I would need to form a map of all the blocks in the play-area and feed that as input layer states instead of the relative location of walls and apples. This method is more thorough and accurate but also resource intensive because of which I have not implemented it.

> This project has been for learning Deep Q Learning and AI through practical exploration
