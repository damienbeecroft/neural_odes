import torch
import torch.nn as nn
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeauralODE(nn.Module): # Neural ODE with one hidden layer
    def __init__(self,ode_dim,hidden_dim = 100,epoch = 20):
        super().__init__()
        self.epoch = epoch # number of steps to take before backpropagating error and training
        self.ode_dim = ode_dim # dimension of the ODE
        self.hidden_dim = hidden_dim # dimension of the hidden layer
        self.A = torch.rand(hidden_dim,ode_dim + 1) # [z_current,tstep] to hidden operator
        self.H = torch.rand(hidden_dim,hidden_dim) # hidden to hidden operator
        self.B = torch.rand(ode_dim,hidden_dim) # hidden to [z_next] operator

    # Compute the neural network approximation of the system, [z_next], at the next time step.
    def forward(self,input,hidden_old):
        hidden = nn.functional.leaky_relu(torch.matmul(self.A,input) + torch.matmul(self.H,hidden_old)) 
        output = torch.matmul(self.B,hidden)
        return (output,hidden)
    
    # Compute the loss at the current step using the sum of the squared error
    def loss(self,estimate,solution):
        return ((estimate - solution)^2).sum()
    
    def adjoint_ode(self):
        vb
    
    # def loss_gradient(self,L):
    #     dLdtheta = 
    
    ###########################################################################################
    # z0        : initial conditions of the ODE system (torch tensor with shape (ode_dim,1))
    # tsteps    : vector of time steps to be taken (torch tensor with shape (N,1))
    # solutions : vector of solutions at the the corresponding time steps (torch tensor with shape (N,1))
    ###########################################################################################
    def train(self,z_0,tsteps,solutions):
        k = int(0)
        N = len(solutions)
        znn_k = z_0 # initialize first approximation as the initial condition
        h_old = torch.zeros(self.hidden_dim) # set the hidden input to the zero vector for the first step
        while (k < N):
            L = 0 # loss accumulator
            for j in range(self.epoch): # accumulate loss in the epoch to then backpropagate on
                Dt = tsteps[k]
                (znn_k,h_old) = self.forward(torch.cat((znn_k,Dt)),h_old)
                L += self.loss(znn_k,solutions[k]) # increment loss
                k += 1
            
            
                