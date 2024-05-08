import torch
from Membrane_PINNs_Polar import Membrane_PINNs
import numpy as np
from Membrane_PINNs_Polar import animate_solution
import time 
import os

model_filename_to_test = "\saved_models\scenario_17_22_27\checkpoint_45000_epochs.pth"
scenraio_id_folder = "\scenario_17_22_27"
path_current_folder = os.path.dirname(os.path.abspath(__file__))
path_model_parameters = "/".join([path_current_folder, model_filename_to_test])

lr = 0.0001

# recreate the model using trained parameters from file
model = Membrane_PINNs(HL_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
checkpoint = torch.load(path_model_parameters)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

epoch = checkpoint['epoch']
total_loss = checkpoint['loss']
eq_loss = checkpoint['eq_loss']
bc_loss = checkpoint['BC_loss']
ic_loss = checkpoint['IC_loss']

print(eq_loss)
print(bc_loss)
print(ic_loss)
print(total_loss)
