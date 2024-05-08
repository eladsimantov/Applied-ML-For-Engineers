"""
This script is to retrain a saved model of type Membrane Pinns. 
It assumes it is has been saved, it continues to train and plots the outputs
"""
import torch
from Membrane_PINNs_Polar import Membrane_PINNs
import numpy as np
from Membrane_PINNs_Polar import animate_solution
import time 
import os


def main():
    # Paths housekeeping - ENTER THE SAVED MODEL FILE TO LOAD 
    model_filename_to_test = "\saved_models\case_2\checkpoint_200000_epochs.pth"
    scenraio_id_folder = "\case_1"
    path_current_folder = os.path.dirname(os.path.abspath(__file__))
    path_model_parameters = "/".join([path_current_folder, model_filename_to_test])
    os.makedirs("/".join([path_current_folder, "outputs"]), exist_ok=True)

    # Set hyperparams
    num_of_epochs = 250001
    lr = 0.0001
    w_eq, w_bc, w_ic = 1, 20, 20

    # recreate the model using trained parameters from file
    model = Membrane_PINNs(HL_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint = torch.load(path_model_parameters)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Set some inputs to the network
    rinitial, rfinal,  = 0.01, 5
    tinitial, tfinal = 0, 20
    theta_initial, theta_final = 0, 2*np.pi
    Nr, Ntheta, Nt = 30, 30, 30
    r, theta, t = model.get_input_tensors(rinitial, rfinal, theta_initial, theta_final, tinitial, tfinal, Nr, Ntheta, Nt)

    epoch = checkpoint['epoch']
    total_loss = checkpoint['loss']
    model.eval()
    model.train()
    start_time = time.time()

    while epoch <= num_of_epochs:
        epoch+=1
        eq_loss, BC_loss, IC_loss = model.compute_loss(r.view(Nr*Ntheta*Nt,1), 
                                                       theta.view(Nr*Ntheta*Nt,1), 
                                                       t.view(Nr*Ntheta*Nt,1), 
                                                       Nr, Ntheta, Nt,
                                                       r_f=rfinal)
        # compute total loss
        total_loss = w_eq*eq_loss + w_bc*BC_loss + w_ic*IC_loss

        # backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch%200 == 0:
            # Save the PINNs model for future use every 5000 epochs (checkpoints)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'BC_loss': BC_loss,
            'IC_loss': IC_loss,
            'eq_loss': eq_loss
            }, "/".join([path_current_folder, 
                         "saved_models", 
                         scenraio_id_folder, 
                         f"checkpoint_{epoch}_epochs.pth"]))
        # skip every 20 epochs before every stat print
        if epoch%20 == 0:
            avg_time_per_epoch = (time.time() - start_time) / epoch
            stats = {"EQ_loss": float(eq_loss),
                     "BC_loss": float(BC_loss),
                     "IC_loss": float(IC_loss),
                     "Total_loss": float(total_loss)}
            minutes_left = round((num_of_epochs - epoch) * avg_time_per_epoch / 60)
            seconds_left = (((num_of_epochs - epoch) * avg_time_per_epoch) % 60)
            losses_msg = ", ".join([str(key) + ": " + f"{value:.12f}" for (key, value) in stats.items()])
            time_stats_msg = f", Estimated time left: {minutes_left} min {seconds_left:.0f} sec"
            print(f"epoch: {epoch}, " + losses_msg + time_stats_msg)
        

    # compute output 
    xi = model.forward(r.view(-1,1), theta.view(-1,1), t.view(-1,1)) # convert tensors into column vectors
    xi_np = xi.detach().numpy() # convert xi into a np array
    xi_reshaped = xi_np.reshape(Nr,Ntheta,Nt) # reshape to fit dimentions

    # create plot for solution
    animate_solution(path_to_folder=path_current_folder, n_epochs=222222,
                        xi=xi_reshaped, Nr=Nr, Ntheta=Ntheta, Nt=Nt, 
                        r_f=rfinal, r_i=rinitial, theta_f=theta_final, theta_i=theta_initial,
                        t_f=tfinal, t_i=tinitial, save_timesteps=True,
                        zlims=[-0.05, 0.05])


if __name__ == "__main__":
    main()