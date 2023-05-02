import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralODE(nn.Module): # Neural ODE with one hidden layer
    ###########################################################################################
    # Initialize variable to be used throughout the struct
    ###########################################################################################
    def __init__(self,ode_dim,hidden_dim = 100,epoch = 20,eps = 1e-2):
        super().__init__()
        self.eps = eps # learning rate of the neural network
        self.epoch = epoch # number of steps to take before backpropagating error and training
        self.ode_dim = ode_dim # dimension of the ODE
        self.hidden_dim = hidden_dim # dimension of the hidden layer
        self.A = torch.rand((hidden_dim,ode_dim),requires_grad=True) # [z_current,tstep] to hidden operator
        self.B = torch.rand((ode_dim,hidden_dim),requires_grad=True) # hidden to [z_next] operator

    ###########################################################################################
    # Compute the neural network approximation of the system, [znn_{k+1}].
    # NOTE: This function only helps us learn autonomous systems
    ###########################################################################################
    # z_k           : The current state (or an approximation of it such as znn_k)
    ###########################################################################################
    def fnn(self,z_k):
        hidden = nn.functional.leaky_relu(torch.matmul(self.A,z_k)) 
        output = torch.matmul(self.B,hidden)
        return output
    
    ###########################################################################################
    # Compute the loss at the current step using the sum of the squared error
    ###########################################################################################
    # znn_k      : The neural network approximation of the state at time step k 
    # z_k        : True solution at time step k 
    ###########################################################################################
    def loss(self,znn_k,z_k):
        return ((znn_k - z_k)^2).sum() 

    ###########################################################################################
    # Compute d(a(t))/dt = (df/dz)^T * a(t)
    ###########################################################################################
    # a         : The Lagrange multiplier (same as the adjoint)
    # _         : Empty slot where the time is input because we are dealing with autonomous ODEs
    # z_k       : The current state (or an approximation of it such as znn_k)
    ###########################################################################################
    def adjoint_ode(self,a,_,z_k):
        f = self.fnn(z_k)
        f.backward(a)
        return -z_k.grad
    
    ###########################################################################################
    # dL/dtheta = int_{t_0}^{t_l} (df/dtheta)^T * a(t). This function computes the points inside
    # the integral
    ###########################################################################################
    # a         : The Lagrange multiplier (same as the adjoint)
    # z_k       : The current state (or an approximation of it such as znn_k)
    ###########################################################################################
    def dLdtheta_points(self,a,z_k):
        f = self.fnn(z_k)
        f.backward(a)
        return ((self.A).grad,(self.B).grad)
    
    ###########################################################################################
    # Function for training the neural ODE network
    ###########################################################################################
    # tsteps    : vector of times at which we have the solution (torch tensor with shape (N,1))
    # z         : vector of solutions at the the corresponding time steps (torch tensor with shape (N,1))
    ###########################################################################################
    def train(self,t,z):
        k = int(0)
        N = len(z)
        epoch = self.epoch
        num_epochs = N//epoch # get number of epochs
        znn = torch.zeros(N,self.ode_dim) # store all the neural net approximations of z
        a = torch.zeros(N,self.ode_dim) # store all the adjoints
        znn[0] = z[0] # initialize first approximation as the initial condition
        for j in range(num_epochs):
            L.detach() # detach the loss function from accumulating gradient data on A, B, and H.
            L = 0 # loss accumulator
            for _ in range(epoch): # accumulate loss in the epoch to then backpropagate on
                (znn[k+1]) = self.fnn(znn[k])
                # L += self.loss(znn[k+1],z[k+1]) # increment loss
                k += 1
            ztemp = (znn[k]).clone().requires_grad_() # temporary variable to track loss
            losstemp = self.loss(ztemp,z[k])
            losstemp.backward() 
            a[k,:] = ztemp.grad # This sets the adjoint at time t_k to be dL/d(z(t_k))
            a[j*epoch:(j+1)*epoch,:] = odeint(self.adjoint_ode,a[k,:],torch.flip(t[j*epoch:(j+1)*epoch],dims = [0])) # NOTE: May be wrong


            # # Naive implementation that we want to avoid.
            # # Here, the graph has to backpropagate through the super graph of the RNN and compute the gradients
            # L.backward()
            # self.H = self.H - self.eps*((self.H).grad)
            # self.A = self.A - self.eps*((self.A).grad)
            # self.B = self.B - self.eps*((self.B).grad)





























#####################################################################################################################################
# OLD FUNCTION DEFINITIONS
#####################################################################################################################################

# The code below requires Dt, but we should not need that since the RNN is learning the derivative function of z

    # ###########################################################################################
    # # Compute the neural network approximation of the system, [znn_{k+1}].
    # ###########################################################################################
    # # input         : This is the augmented input [znn_k, Dt]
    # # hidden_old    : The hidden layer from the previous iteration of the neural ODE
    # ###########################################################################################
    # def fnn(self,input,hidden_old):
    #     hidden = nn.functional.leaky_relu(torch.matmul(self.A,input) + torch.matmul(self.H,hidden_old)) 
    #     output = torch.matmul(self.B,hidden)
    #     return (output,hidden)

# In this implementation I useed a recurrent neural net, however you can't do adaptive time stepping with this.
# So, I decided take the recurrent part out.

    # ###########################################################################################
    # # Compute the neural network approximation of the system, [znn_{k+1}].
    # # NOTE: This function only helps us learn autonomous systems
    # ###########################################################################################
    # # z_k           : The current state (or an approximation of it such as znn_k)
    # # hidden_old    : The hidden layer from the previous iteration of the neural ODE
    # ###########################################################################################
    # def fnn(self,z_k,hidden_old):
    #     hidden = nn.functional.leaky_relu(torch.matmul(self.A,z_k) + torch.matmul(self.H,hidden_old)) 
    #     output = torch.matmul(self.B,hidden)
    #     return (output,hidden)
            
                