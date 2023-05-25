# Import Libraries
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

# Enable Interactive Mode
plt.ion()

def plot(scores, mean_scores, current_mean_scores, minY, maxY):
    # Setup IPython
    display.clear_output(wait=True)
    display.display(plt.figure("Learning Graph"))
    # Plot graph
    plt.clf()
    plt.title('Progress of Machine Learning Algorithm')
    plt.xlabel('AI Generation')
    plt.ylabel('Points Scored by AI')
    plt.yticks(np.arange(minY, maxY + 10, 10))
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(current_mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(current_mean_scores)-1, current_mean_scores[-1], str(current_mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

# TODO: I can try to make multiple graphs to clean up the clutter.
#  I intended on making a time survived graph but the snake infini-looping needs to be resolved first.
