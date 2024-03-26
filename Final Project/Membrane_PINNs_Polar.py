import torch
import torch.nn as nn
from torch.autograd import grad
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.special import bessel_j0 as J0
from scipy.special import jn_zeros
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time
import os

class ffm(nn.Module):
    def __init__(self, in_dim, out_dim, std_dev = 2.0):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(out_dim, in_dim) * std_dev) 

    def forward(self, x):
        return torch.cos(F.F.linear(x, self.omega))


class Membrane_PINNs(nn.Module):    
    def __init__(self, in_dim=3, HL_dim=16, out_dim=1, activation=nn.Tanh(), use_ffm=False):
        """
        Parameters
        -------------
        in_dim: the input dimensions - number of independant variables
        HL_dim: the width of the network
        out_dim: the output dimensions - number of dependant variables
        activation: The activation function you wish to use in the network - the default is nn.Tanh()
        use_ffm: A bool for deciding to use FFM in input or not.
        diff_coeff: The diffusion coefficient used in the PDE
        """
        super().__init__()

        # define the network architecture
        network = [ffm(in_dim, HL_dim)] if use_ffm else [nn.Linear(in_dim, HL_dim), activation]
        network += [
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, out_dim)
                   ]
        
        # define the network using sequential method
        self.xi = nn.Sequential(*network) 


    def forward(self, r, theta, t):
        return self.xi(torch.cat((r, theta, t), 1))
    

    def compute_loss(self, r, theta, t,  Nr, Ntheta, Nt, r_f):
        """
        This is the physics part
        """
        r.requires_grad=True
        theta.requires_grad=True
        t.requires_grad=True
        xi = self.xi(torch.cat((r, theta, t), 1))

        # compute PDE derivatives using auto grad
        xi_r = grad(xi, r, grad_outputs=torch.ones_like(xi), create_graph=True)[0] # we need to specify the dimension of the output array
        xi_rr = grad(xi_r, r, grad_outputs=torch.ones_like(xi_r), create_graph=True)[0]
        rxi_rr_over_r = xi_rr + xi_r / r

        xi_theta = grad(xi, theta, grad_outputs=torch.ones_like(xi), create_graph=True)[0]
        xi_ttheta = grad(xi_theta, theta, grad_outputs=torch.ones_like(xi_theta), create_graph=True)[0]
        xi_ttheta_over_r2 = xi_ttheta / r**2

        xi_t = grad(xi, t, grad_outputs=torch.ones_like(xi), create_graph=True)[0]
        xi_tt = grad(xi_t, t, grad_outputs=torch.ones_like(xi_t), create_graph=True)[0]

        # set a loss function to apply to each of the physics residuals (PDE, IC, BC)
        loss_fun = nn.MSELoss()

        # compute the PDE residual loss - only using here solution for initial condition, no external force
        residual = xi_tt - 1**2 * (rxi_rr_over_r) # case axissymetric
        # residual = xi_tt - 1**2 * (xi_ttheta_over_r2 + rxi_rr_over_r)
        pde_loss = loss_fun(residual, torch.zeros_like(residual))

        # compute the BC loss - periodic and with a fixed boundary layer
        xi_reshaped = xi.view(Nr, Ntheta, Nt) # [Nr*Ntheta*Nt, 1] -> [Nr, Ntheta, Nt]
        xi_r_reshaped = xi_r.view(Nr, Ntheta, Nt) # [Nr*Ntheta*Nt, 1] -> [Nr, Ntheta, Nt]
        xi_theta_reshaped = xi_theta.view(Nr, Ntheta, Nt)

        # The BC loss is a combination of the following:
        # 1. xi at theta=0 and theta=2pi are equal
        # 2. xi_theta at theta=0 and theta=2pi are equal
        # 3. xi are r=R is zero
        # 4. xi_r at r->0 is zero
        bc_loss = loss_fun(xi_reshaped[:,0,:], xi_reshaped[:,Ntheta-1,:]) \
                + loss_fun(xi_theta_reshaped[:,0,:], xi_theta_reshaped[:,Ntheta-1,:]) \
                + loss_fun(xi_reshaped[Nr-1,:,:], torch.zeros_like(xi_reshaped[Nr-1,:,:])) \
                # + loss_fun(xi_r_reshaped[0,:,:], torch.zeros_like(xi_r_reshaped[0,:,:])) \
        
        # compute the IC loss
        r_reshaped = r.view(Nr, Ntheta, Nt)
        theta_reshaped = theta.view(Nr, Ntheta, Nt)
        
        # set an IC for the first mode of vibration
        a_01 = jn_zeros(0, 1)[-1] # get correct alpha for first mode
        xi_initial = 0.05 * J0(r_reshaped[:,:,0] * a_01 / r_f)
        ic_loss = loss_fun(xi_initial, xi_reshaped[:,:,0])
    
        return pde_loss, bc_loss, ic_loss
    

    def get_input_tensors(self, r_i, r_f, theta_i, theta_f, t_i, t_f, Nr, Ntheta, Nt):
        dr = (r_f - r_i) / (Nr-1)
        dtheta = (theta_f - theta_i) / (Ntheta-1)
        dt = (t_f - t_i) / (Nt-1)
        r = torch.zeros(Nr, Ntheta, Nt)
        theta = torch.zeros(Nr, Ntheta, Nt)
        t = torch.zeros(Nr, Ntheta, Nt)
        print(f"Creating input tensors Nr={Nr}, Ntheta={Ntheta}, Nt={Nt}")
        for i in range(Nr):
            for j in range(Ntheta):
                for k in range(Nt):
                    r[i,j,k] = r_i + dr * i
                    theta[i,j,k] = theta_i + dtheta * j
                    t[i,j,k] = t_i + dt * k
        print("Done creating input tensors")
        return r, theta, t



def animate_solution(path_to_folder, xi, r_i, r_f, theta_i, theta_f, t_i, t_f, Nr, Ntheta, Nt, n_epochs, save_timesteps=False, zlims=[-0.005, 0.005]):
    """
    This function will convert the solution from xi as function of r theta and t to a 3d representation.
    It will create and save an animation of the displacement of the membrane in time along with all the timesteps pictures.
    """
               
    # Create the mesh in polar coordinates and compute corresponding Z.
    Radius, Angle = np.meshgrid(np.linspace(r_i, r_f, Nr), 
                                np.linspace(0, 2*np.pi, Ntheta), 
                                indexing='ij') # The indexing 'ij' was nasty. It makes sure the plotting is by the Z indexes. 
    X, Y = Radius*np.cos(Angle), Radius*np.sin(Angle)
    
    # Prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize animation frame
    def init():
        ax.clear()
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])
        ax.set_zlim(zlims)

    # Update function for animation
    def update(frame):
        Z = xi[:,:,frame] # because we used indexing 'ij' instead of 'xy' in the meshgrid this should work.
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_zlim(zlims)
        return fig

    # Create and save animation
    time_str = time.strftime("%H_%M_%S", time.localtime())
    ani = FuncAnimation(fig, update, frames=Nt, init_func=init, blit=False)
    plt.show()
    animation_id = f"{time_str}_time_{n_epochs}_epochs_{Nt}_timesteps"
    ani.save("/".join([path_to_folder, "outputs", animation_id + ".gif"])) 
    plt.close()

    # ---------------------- #

    if save_timesteps:
        # create a directory for the animation timesteps
        os.makedirs("/".join([path_to_folder, "outputs", animation_id]))

        # Plot the surface at all Nt states and save pictures.
        for timestep in range(Nt):
            Z = xi[:,:,timestep]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis')
            ax.set_zlim(zlims)
            plt.savefig("/".join([path_to_folder, "outputs", animation_id, f"timestep {timestep}"]))
            plt.close()
    return

    
def main():
    path_current_folder = os.path.dirname(os.path.abspath(__file__))
    os.makedirs("/".join([path_current_folder, "outputs"]), exist_ok=True)

    # Set domain bounds and resolution
    rinitial, rfinal,  = 0.01, 5
    tinitial, tfinal = 0, 20
    theta_initial, theta_final = 0, 2*np.pi
    Nr, Ntheta, Nt = 20, 20, 20
    
    # Set hyperparams
    num_of_epochs = 500
    lr = 0.001
    w_eq, w_bc, w_ic = 5, 20, 20

    # create PINNs model
    model = Membrane_PINNs(HL_dim=5)

    # initiallize input parameters as tensors
    r, theta, t = model.get_input_tensors(rinitial, rfinal, theta_initial, theta_final, tinitial, tfinal, Nr, Ntheta, Nt)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses_history = np.zeros((num_of_epochs, 4))
    start_time = time.time()

    for epoch in range(num_of_epochs):
        eq_loss, BC_loss, IC_loss = model.compute_loss(r.view(Nr*Ntheta*Nt,1), 
                                                       theta.view(Nr*Ntheta*Nt,1), 
                                                       t.view(Nr*Ntheta*Nt,1), 
                                                       Nr, Ntheta, Nt,
                                                       r_f=rfinal)
        # compute total loss
        total_loss = w_eq*eq_loss + w_bc*BC_loss + w_ic*IC_loss

        # save date for losses history
        losses_history[epoch, 0] = total_loss
        losses_history[epoch, 1] = eq_loss
        losses_history[epoch, 2] = BC_loss
        losses_history[epoch, 3] = IC_loss

        # backward pass
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # skip every 20 epochs before every stat print
        if epoch%20 == 0 and epoch >= 1:
            avg_time_per_epoch = (time.time() - start_time) / epoch
            stats = {"EQ_loss": float(eq_loss),
                     "BC_loss": float(BC_loss),
                     "IC_loss": float(IC_loss),
                     "Total_loss": float(total_loss)}
            minutes_left = round((num_of_epochs - epoch) * avg_time_per_epoch / 60)
            seconds_left = (((num_of_epochs - epoch) * avg_time_per_epoch) % 60)
            losses_msg = ", ".join([str(key) + ": " + f"{value:.8f}" for (key, value) in stats.items()])
            time_stats_msg = f", Estimated time left: {minutes_left} min {seconds_left:.0f} sec"
            print(f"epoch: {epoch}, " + losses_msg + time_stats_msg)
        
        # Add a stop criteria
        if total_loss <= 0.00000001:
            print(f"epoch: {epoch}, loss: {total_loss:.8f}")
            print("Reached stop criterion")
            break

        
    # plot loss history
    plt.plot(losses_history)
    plt.title("Losses History")
    plt.legend(["Total Loss", "PDE Loss", "BC Loss", "IC Loss"])
    plt.grid(visible=True)
    plt.savefig("/".join([path_current_folder, "outputs", f"Loss after {epoch+1} epochs"]))
    plt.close()

    # Test the PINNs
    xi = model.forward(r.view(-1,1), theta.view(-1,1), t.view(-1,1)) # convert tensors into column vectors
    xi_np = xi.detach().numpy() # convert xi into a np array
    xi_reshaped = xi_np.reshape(Nr,Ntheta,Nt) # reshape to fit dimentions

    # xi_reshaped[Nr-1,:,:] = 0 # force BC to zero after the solution is computed

    animate_solution(path_to_folder=path_current_folder, n_epochs=num_of_epochs,
                     xi=xi_reshaped, Nr=Nr, Ntheta=Ntheta, Nt=Nt, 
                     r_f=rfinal, r_i=rinitial, theta_f=theta_final, theta_i=theta_initial,
                     t_f=tfinal, t_i=tinitial, 
                     zlims=[-0.05, 0.05])
    
    # Save the PINNs model for future use
    torch.save(model.state_dict(), "/".join([path_current_folder, "saved_model_parameters.pth"]))
    return
    


if __name__ == "__main__":
    main()
    print("Finished running main without exceptions")

