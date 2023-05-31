import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Code adapted from Ricky Chen's ode_demo.py, found at https://github.com/rtqichen/torchdiffeq/tree/master/examples

name = 'backprop_cubic_nn2_1_100000_1000' # name of files/folders ouput

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=100000) # data size was originally 1000
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = 'cpu'

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):
    '''
    True ODE:
    y' = -0.1 * x^3 + 2.0 * y^3
    x' = -2.0 * x^3 + -0.1 * y^3
    '''
    def forward(self, t, y):
        return torch.mm(y**3, true_A)

# Get true solution data of the ODE using the Lambda() class
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')

# Create batches for training
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# Visualization code
if args.viz:
    # makedirs('png')
    makedirs(name)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    with torch.no_grad():
        if args.viz:

            ax_traj.cla()
            ax_traj.set_title('Trajectories')
            ax_traj.set_xlabel('t')
            ax_traj.set_ylabel('x,y')
            ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
            ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
            ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
            ax_traj.set_ylim(-2, 2)
            ax_traj.legend()

            ax_phase.cla()
            ax_phase.set_title('Phase Portrait')
            ax_phase.set_xlabel('x')
            ax_phase.set_ylabel('y')
            ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
            ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
            ax_phase.set_xlim(-2, 2)
            ax_phase.set_ylim(-2, 2)

            ax_vecfield.cla()
            ax_vecfield.set_title('Learned Vector Field')
            ax_vecfield.set_xlabel('x')
            ax_vecfield.set_ylabel('y')

            y, x = np.mgrid[-2:2:21j, -2:2:21j]
            dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
            mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
            dydt = (dydt / mag)
            dydt = dydt.reshape(21, 21, 2)

            ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
            ax_vecfield.set_xlim(-2, 2)
            ax_vecfield.set_ylim(-2, 2)

            fig.tight_layout()
            plt.savefig(name + '/{:03d}'.format(itr))
            plt.draw()
            plt.pause(0.001)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # The following three code segments define the operations that are applied to the input.
        # Two out of three of the segments should be commented out.

        # Train the 2x2 coefficient matrix of the cubic oscillator. Should converge to true_A
        # self.net = nn.Sequential(
        #     nn.Linear(2,2)
        # )

        # # Train a neural net with a single hidden layer. This one does not converge well
        # self.net = nn.Sequential(
        #     nn.Linear(2, 50),
        #     nn.Tanh(),
        #     nn.Linear(50, 2),
        # )

        # Train a neural net with a two hidden layers. This one does not converge well
        
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # return self.net(y**3)
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    # a_tols = torch.tensor([10**(-k) for k in range(1,6)])
    data1 = torch.load(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\cubic\output\data\backprop_cubic_nn2_1_100000_1000.pt")

    # loss_tracker = torch.tensor([]) # records the loss each epoch


    # num_its = 100000 # number of epochs to run
    # test_freq = 1000 # how often to sample the output of the test

    ii = 0

    func = ODEFunc().to(device)
    func.load_state_dict(data1['func'])
    
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    # end = time.time()
    pred_y = odeint(func, true_y0, t)
    loss = torch.mean(torch.abs(pred_y - true_y))
    # loss_tracker = torch.cat((loss_tracker,loss.reshape(1)),dim=0) # concatenate to the loss_tracker
    # print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
    visualize(true_y, pred_y, func, ii) # visualise if --viz option was set
    end = time.time()

    # time_meter = RunningAverageMeter(0.97)
    
    # loss_meter = RunningAverageMeter(0.97)

    # for itr in range(1, num_its + 1):
    #     optimizer.zero_grad()
    #     batch_y0, batch_t, batch_y = get_batch()
    #     pred_y = odeint(func, batch_y0, batch_t).to(device)
    #     loss = torch.mean(torch.abs(pred_y - batch_y))
    #     loss.backward()
    #     optimizer.step()

    #     time_meter.update(time.time() - end)
    #     loss_meter.update(loss.item())

    #     if itr % test_freq == 0:
    #         with torch.no_grad():
    #             pred_y = odeint(func, true_y0, t)
    #             loss = torch.mean(torch.abs(pred_y - true_y))
    #             loss_tracker = torch.cat((loss_tracker,loss.reshape(1)),dim=0) # concatenate to the loss_tracker
    #             print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
    #             # visualize(true_y, pred_y, func, ii) # visualise if --viz option was set
    #             end = time.time()

    #             save = {
    #                 'optim': optimizer.state_dict(), # save optimizer
    #                 'func': func.state_dict(), # save trained network
    #                 'loss_tracker': loss_tracker, # save the loss tracking
    #                 'time_meter': time_meter, # time meter with weight 0.97
    #                 'epochs': range(0,num_its,test_freq), # save when the loss was recorded
    #                 'num_its': num_its,
    #                 'test_freq': test_freq,
    #                 'itr': itr,
    #             }
    #             torch.save(save, name + '.pt')
    #             ii += 1
        
        # end = time.time()

    # save = {
    #     'optim': optimizer.state_dict(), # save optimizer
    #     'func': func.state_dict(), # save trained network
    #     'loss_tracker': loss_tracker, # save the loss tracking
    #     'time_meter': time_meter, # time meter with weight 0.97
    #     'epochs': range(0,num_its,test_freq), # save when the loss was recorded
    #     'num_its': num_its,
    #     'test_freq': test_freq
    # }
    # torch.save(save, name + '.pt')
