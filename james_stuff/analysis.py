import argparse
from adjoint_calculation import adjoint_calculate
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE adjoint demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
args = parser.parse_args()

device = 'cpu'
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.dumb_param = nn.Parameter(torch.tensor([1.0]))

    def forward(self, t, y):
        return torch.mm(y**3, true_A) * self.dumb_param

def visualize(y, ax):
    ax.cla()
    ax.set_title('Phase Portrait')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
    ax.plot(y[:, 0, 0], y[:, 0, 1], 'b-')

    # Color scatter points over time.
    T = y.shape[0]
    nper = 10
    for t in range(0, T, nper):
        c = [t / float(T), 0., 0.]
        plt.scatter(y[t:t+nper, 0, 0], y[t:t+nper, 0, 1], color = c)

if __name__ == '__main__':
    tol = 1e-4
    ode = Lambda()
    true_y = odeint_adjoint(ode, true_y0, t, method=args.method, rtol = tol, atol = tol)

    loss = torch.mean(true_y)
#    loss.backward()
        
    record = adjoint_calculate(t, true_y, ode, args.method, tol)

    # Record is a list of augmented states. 
    # Each augmented state is a list [<UNUSED>, state, adjoint, running grad].
    states = np.array([a[1].detach().numpy() for a in record])
    adjoints = np.array([a[2].detach().numpy() for a in record])
    grads = np.array([a[3].detach().numpy() for a in record])
    print(states.shape, adjoints.shape, grads.shape, true_y.shape)

    # Reverse temporally since these were simulated backwards in time.
    states = states[::-1]
    adjoints = adjoints[::-1]
    grads = grads[::-1]

    plt.figure(figsize=(15,6))
    ax = plt.subplot(131)
    visualize(states, ax)
    plt.title('ODE Flow')
    ax = plt.subplot(132)
    visualize(adjoints, ax)
    plt.title('Adjoint Flow')
    ax = plt.subplot(133)
    plt.plot(t, grads)
    plt.title('Running Gradient')
    plt.tight_layout()
    plt.show()
