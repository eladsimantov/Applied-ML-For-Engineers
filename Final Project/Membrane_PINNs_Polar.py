import torch
import torch.nn as nn
from torch.autograd import grad
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt


class ffm(nn.Module):
    def __init__(self, in_dim, out_dim, std_dev = 2.0):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(out_dim, in_dim) * std_dev) 

    def forward(self, x):
        return torch.cos(F.F.linear(x, self.omega))


class Membrane_PINNs(nn.Module):    
    def __init__(self, in_dim=3, HL_dim=32, out_dim=1, activation=nn.Tanh(), use_ffm=False, R=1, c=1):
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
        network = [ffm(in_dim, HL_dim)] if use_ffm else [nn.Linear(in_dim, HL_dim)]
        network += [
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, HL_dim), activation,
                   nn.Linear(HL_dim, out_dim)
                   ]
        
        # define the network using sequential method
        self.xi = nn.Sequential(*network) 
        self.c2 = c**2
        self.R = R


    def set_inputs(Nr, Ntheta, Nt, r_final, t_final):
        r_initial = 0
        theta_initial, theta_final = 0, 2*np.pi()
        t_initial = 0
        dr = (r_final - r_initial) / (Nr-1)
        dtheta = (theta_final - theta_initial) / (Ntheta-1)
        dt = (t_final - t_initial) / (Nt-1)

        # initiallize input parameters as tensors
        r = torch.zeros(Nr, Ntheta, Nt)
        theta = torch.zeros(Nr, Ntheta, Nt)
        t = torch.zeros(Nr, Ntheta, Nt)
        for i in range(Nr):
            for j in range(Ntheta):
                for k in range(Nt):
                    r[i,j,k] = r_initial + dr * i
                    theta[i,j,k] = theta_initial + dtheta * j
                    t[i,j,k] = t_initial + dt * k
        return r, theta, t


    def forward(self, r, theta, t):
        return self.xi(torch.cat((r, theta, t), 1))
    

    def compute_loss(self, r, theta, t,  Nr, Ntheta, Nt):
        """
        This is the physics part really
        """
        r.requires_grad=True
        theta.requires_grad=True
        t.requires_grad=True
        xi = self.xi(torch.cat((r,theta, t), 1))

        # compute PDE derivatives using auto grad
        xi_r = grad(xi, r, grad_outputs=torch.ones_like(xi), create_graph=True)[0] # we need to specify the dimension of the output array
        rxi_r = r * xi_r
        rxi_rr = grad(rxi_r, r, grad_outputs=torch.ones_like(rxi_r), create_graph=True)[0]
        rxi_rr_over_r = rxi_rr / r

        xi_theta = grad(xi, theta, grad_outputs=torch.ones_like(xi))
        xi_ttheta = grad(xi_theta, theta, grad_outputs=torch.ones_like(xi_theta))
        xi_ttheta_over_r2 = xi_ttheta / r*2

        xi_t = grad(xi, t, grad_outputs=torch.ones_like(xi))
        xi_tt = grad(xi_t, t, grad_outputs=torch.ones_like(xi))

        # set a loss function to apply to each of the physics residuals (PDE, IC, BC)
        loss_fun = nn.MSELoss()

        # compute the PDE residual loss - only using here solution for initial condition, no external force
        residual = xi_tt - self.c2 * (xi_ttheta_over_r2 + rxi_rr_over_r)
        pde_loss = loss_fun(residual, torch.zeros_like(residual))

        # compute the BC loss - periodic and with a fixed boundary layer
        xi_reshaped = xi.view(Nr, Ntheta, Nt) # [Nr*Ntheta*Nt, 1] -> [Nr, Ntheta, Nt]
        xi_r_reshaped = xi_r.view(Nr, Ntheta, Nt) # [Nr*Ntheta*Nt, 1] -> [Nr, Ntheta, Nt]
        bc_loss = loss_fun(xi_reshaped[Nr-1,:,:], torch.zeros_like(xi_reshaped[Nr-1,:,:])) \
                + loss_fun(xi_r_reshaped[0,:,:], xi_r_reshaped[Nr-1,:,:]) # derivative at r=R should be symmetric
        
        # compute the IC loss
        r_reshaped = r.view(Nr, Ntheta, Nt)
        theta_reshaped = theta.view(Nr, Ntheta, Nt)
        #!TODO set proper IC
        xi_initial = torch.cos(2 * np.pi * r_reshaped[:,:,0]) # TBD
        ic_loss = loss_fun(xi_initial, xi_reshaped[:,:,0])
    
        return pde_loss, bc_loss, ic_loss
    


def main():
    Nr, Ntheta, Nt = 128, 128, 128
    rfinal, tfinal = 1, 32
    r_initial, tinitial = 0, 0
    num_of_epochs = 20000
    lr = 0.001
    w_eq, w_bc, w_ic = 1, 20, 20
    theta_initial, theta_final = 0, 2*np.pi
    dr = (rfinal - r_initial) / (Nr-1)
    dtheta = (theta_final - theta_initial) / (Ntheta-1)
    dt = (tfinal - tinitial) / (Nt-1)

    # initiallize input parameters as tensors
    r = torch.zeros(Nr, Ntheta, Nt)
    theta = torch.zeros(Nr, Ntheta, Nt)
    t = torch.zeros(Nr, Ntheta, Nt)
    for i in range(Nr):
        for j in range(Ntheta):
            for k in range(Nt):
                r[i,j,k] = r_initial + dr * i
                theta[i,j,k] = theta_initial + dtheta * j
                t[i,j,k] = tinitial + dt * k

    model = Membrane_PINNs()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses_history = np.zeros((num_of_epochs, 4))
    for epoch in range(num_of_epochs):

        # compute various losses
        eq_loss, BC_loss, IC_loss = model.compute_loss(r.view(-1,1), 
                                                       theta.view(-1,1), 
                                                       t.view(-1,1), 
                                                       Nr, Ntheta, Nt)

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

        # skip by 500 epochs before every print
        if not epoch%500:
            print(f"epoch: {epoch}, loss: {total_loss}")
        
        # plot solutions by the end of [10000, 20000] epochs
        if epoch+1 in [10000, 20000]:
            xi = model.forward(r.view(-1,1), theta.view(-1,1), t.view(-1,1)) # convert x tensor into a column vector
            xi_np = xi.detach().numpy() # convert into a np array
            xi_reshaped = xi_np.reshape(Nr,Ntheta,Nt)
            #!TODO make a visual meshgrid for membrane vibrations    

    plt.plot(losses_history)
    plt.title("Losses History")
    plt.legend(["Total Loss", "PDE Loss", "BC Loss", "IC Loss"])
    return

if __name__ == "__main__":
    main()
