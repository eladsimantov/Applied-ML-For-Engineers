import torch
import torch.nn as nn
from torch.autograd import grad
import torch.functional as F
import numpy as np

class ffm(nn.Module):
    def __init__(self, in_dim, out_dim, std_dev = 2.0):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(out_dim, in_dim) * std_dev) 

    def forward(self, x):
        return torch.cos(F.F.linear(x, self.omega))
    
class PINN(nn.Module):
    """
    This class will ...
    """
    def __init__(self, 
                 de_order:int, 
                 out_dims:int, 
                 domain:np.array, 
                 activation:torch.nn.modules.activation, 
                 HL_dims:np.array,
                 ffm=False,
                 ffm_std=0.0
                 ):
        """
        Parameters
        -----------
        1. de_order: The order of the differential equation function for the residual. It defines the ouptut dimentions  
        2. out_dims: Number of output variables  
        3. domain: An np.array with bounds of each dimention. It defines the input dimentions e.g. np.array([[0, 1][0, 1]]) has 2 inputs  
        4. activation: An activation function from torch.nn.modules.activation class to be applied in hidden layers  
        5. HL_dims: A list with the number of units per hidden layer for the sequential network. 
        6. ffm: Include Fourier Feature Mapping - Boolean set to False by default
        7. ffm_std: Standard Deviation for the FFM layer
        """
        super().__init__()
        in_dims = domain.shape[0] # extract dimentions from domain shape
        self.in_dims = in_dims
        self.width = HL_dims
        self.de_order = de_order
        self.domain = domain
        self.function_defined = False
        self.activation_function = activation
        self.io_structure = [out_dims] + [in_dims for _ in de_order]

        # define the network - start, middle, end
        start = [ffm(in_dims, HL_dims[0], ffm_std)] if ffm else [nn.Linear(in_dims, HL_dims[0]), activation]
        middle = [[nn.Linear(HL_dims[_])] + [activation] for _ in range(len(HL_dims))]
        end = [nn.Linear(HL_dims[-1], out_dims)]
        self.outputs = nn.Sequential(start + middle + end)
        return
    

    def set_differential_equation(self, f:function):
        parameter_matrix = np.zeros(self.io_structure)
        try:
            f(parameter_matrix)
        except:
            "Mismatch between your input dimentions, output dimentions or differential equation order"
        self.function_defined = True
        self.f = f
        return 


    def forward(self, X):
        return self.outputs(torch.cat(*X, 1))
        

    def compute_de_loss(self, X:list, Y:list, loss_function=nn.MSELoss):
        """
        Calculate the loss of the Differential equation given 
        X: The inputs of the network - (List of Torch Tensors)
        Y: The outputs of the network - (List of Torch Tensors)
        loss_function: The type of loss function - Set to MSELoss by default
        """
        parameter_matrix = 
        if self.function_defined:
            return self.f(X)
        raise "Differential Equation Function not defined yet"
    

    def train(self, epochs, optimizer, lr):
        return


obj = PINN(1, 2, np.array([[0, 1][0, 1]]), nn.Tanh(), [64, 128, 64])

obj.set_differential_equation()
obj.set_boundary_conditions()
obj.set_initial_condition()
obj.train()
obj.get_loss_history()
obj.