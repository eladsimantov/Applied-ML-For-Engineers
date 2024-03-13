# import torch
# import torch.nn as nn
# from torch.autograd import grad
# import torch.functional as F
# import numpy as np

# class PINN(nn.Module):
#     """
    
#     """
#     def __init__(self, f:function, domain:np.array , activation:torch.nn.modules.activation, HL_dims:np.array):
#         """
#         Parameters
#         -----------
#         f: The differential equation function for the residual. It defines the ouptut dimentions
#         domain: An np.array with bounds of each dimention. It defines the input dimentions e.g. np.array([[0, 1][0, 1]]) has 2 inputs
#         activation: An activation function from torch.nn.modules.activation class to be applied in hidden layers
#         HL_dims: An np.array with the number of units per input layer
#         """
#         super().__init__()
#         in_dims = domain.shape[0]

#         num_of_args = lambda *args: len(args)

#         out_dims = lambda *args: f(args)
#         pass

#     def forward():
#         pass

#     def get_equation_loss(self, X):

#         pass


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net(ax, layer_sizes):
    """
    Draw a neural network cartoon using matplotilb.
    
    :param ax: matplotlib.axes.Axes, the axes on which to plot the cartoon (get e.g. by plt.gca())
    :param layer_sizes: List of layer sizes, including input and output layer
    """
    n_layers = len(layer_sizes)
    v_spacing = (1. / max(layer_sizes)) * .8
    h_spacing = (1. / (n_layers - 1)) * .8

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = 1 - (v_spacing * layer_size) / 2.
        for m in range(layer_size):
            circle = patches.Circle((n * h_spacing, layer_top + m * v_spacing), v_spacing / 4.,
                                    edgecolor='k', facecolor='w', zorder=4)
            ax.add_patch(circle)
            # Annotation for the first layer
            if n == 0:
                ax.annotate(f'Input\nFeature {m+1}', (n * h_spacing, layer_top + m * v_spacing), xytext=(-10, 5),
                             textcoords='offset points', ha='right', va='bottom', fontsize=8)
            # Annotation for the last layer
            elif n == len(layer_sizes) - 1:
                ax.annotate('Output', (n * h_spacing, layer_top + m * v_spacing), xytext=(10, 5),
                             textcoords='offset points', ha='left', va='center', fontsize=8)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = 1 - (v_spacing * layer_size_a) / 2.
        layer_top_b = 1 - (v_spacing * layer_size_b) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = patches.FancyArrowPatch((n * h_spacing, layer_top_a + m * v_spacing),
                                               ((n + 1) * h_spacing, layer_top_b + o * v_spacing),
                                               connectionstyle="arc3,rad=.1", arrowstyle='->', color="k", lw=0.5)
                ax.add_patch(line)

layer_sizes = [2, 10, 1]  # Input layer, 4 hidden layers, output layer
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, layer_sizes)
plt.show()


