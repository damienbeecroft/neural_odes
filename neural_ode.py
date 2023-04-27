import torch
import torch.nn as nn
import numpy as np

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
        self.A = torch.rand((hidden_dim,ode_dim + 1),requires_grad=True) # [z_current,tstep] to hidden operator
        self.H = torch.rand((hidden_dim,hidden_dim),requires_grad=True) # hidden to hidden operator
        self.B = torch.rand((ode_dim,hidden_dim),requires_grad=True) # hidden to [z_next] operator

    ###########################################################################################
    # Compute the neural network approximation of the system, [znn_{k+1}].
    # NOTE: This function only helps us learn autonomous systems
    ###########################################################################################
    # z_k           : The current state (or an approximation of it such as znn_k)
    # hidden_old    : The hidden layer from the previous iteration of the neural ODE
    ###########################################################################################
    def fnn(self,z_k,hidden_old):
        hidden = nn.functional.leaky_relu(torch.matmul(self.A,z_k) + torch.matmul(self.H,hidden_old)) 
        output = torch.matmul(self.B,hidden)
        return (output,hidden)
    
    ###########################################################################################
    # Compute the loss at the current step using the sum of the squared error
    ###########################################################################################
    # znn_k      : The neural network approximation of the state at time step k 
    # z_k        : True solution at time step k 
    ###########################################################################################
    def loss(self,znn_k,z_k):
        return ((znn_k - z_k)^2).sum() 

    ###########################################################################################
    # Compute d(lambda(t))/dt = (df/dz)^T * lambda(t)
    ###########################################################################################
    # z_k       : The current state (or an approximation of it such as znn_k)
    # l         : The Lagrange multiplier
    ###########################################################################################
    def adjoint_ode(self,z_k,hidden,l):
        f,_ = self.fnn(z_k,hidden)
        dldt = f.backward(l)
        dldt.detach()
        return dldt
    
    # def loss_gradient(self,L):
    #     dLdtheta =    
    
    ###########################################################################################
    # Function for training the neural ODE network
    ###########################################################################################
    # tsteps    : vector of time steps to be taken (torch tensor with shape (N,1))
    # z         : vector of solutions at the the corresponding time steps (torch tensor with shape (N,1))
    ###########################################################################################
    def train(self,tsteps,z):
        k = int(0)
        N = len(z)
        znn = torch.zeros(N,self.ode_dim) # store all the neural net approximations of z
        l = torch.zeros(N,self.ode_dim) # store all the adjoints
        znn[0] = z[0] # initialize first approximation as the initial condition
        h_old = torch.zeros(self.hidden_dim) # set the hidden input to the zero vector for the first step
        while (k < N):
            L.detach() # detach the loss function from accumulating gradient data on A, B, and H.
            L = 0 # loss accumulator
            for _ in range(self.epoch): # accumulate loss in the epoch to then backpropagate on
                (znn[k+1],h_old) = self.fnn(znn[k],h_old)
                # L += self.loss(znn[k+1],z[k+1]) # increment loss
                k += 1
            ztemp = (znn[k]).clone().requires_grad_() # temporary variable to track loss
            losstemp = self.loss(ztemp,z[k])
            losstemp.backward() 
            l[k,:] = ztemp.grad # This sets the adjoint at time t_k to be dL/d(z(t_k))
            for j in range(k-1,k-self.epoch-1,-1): # NOTE: This may need to be adjusted. I am not sure that the increments are exactly right
                l[j,:] = l[j+1,:] - 


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

# I was trying to do epochs in the code, so I wanted to have an inner for loop
            
                