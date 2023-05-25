# SnakeAI
## The classic game of snake, but it is controlled by a Deep RL algorithm

### Objective
> This project has been for learning about Deep Q Learning, Artificial Neural Networks, and AI through practical exploration
- Teaching AI how to play a videogame, such as 'Snake', was a double edged sword for me. 
  - Not only did I get an efficient way to understand how AI algorithms work and learn, but I also got to better understand the science behind optimising results in a video game.
- 

### Resources
> The entirety of this project is written in Python 3.10
I have created the game of Snake using the PyGame library, and implented the ANN using PyTorch.

#### The algorithm is currently only learning for ~500 runs due to a hidden layer constraint and an error in the algorithm that causes the snake to loop and collide into itself. 
#### To solve that, I would need to form a map of all the blocks in the play-area and feed that as input layer states instead of the relative location of walls and apples. 
#### This method is more thorough and accurate but also resource intensive because of which I have not yet implemented it.
