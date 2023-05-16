import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from adjoint_calculation import adjoint_calculate
import matplotlib.pyplot as plt
from torchdiffeq import odeint

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
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

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

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
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


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

def plot_flow(y, ax):
    ax.cla()
    ax.set_title('Phase Portrait, red = start, blue = end')
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
    ax.plot(y[:, 0, 0], y[:, 0, 1], 'b-')

    # Color scatter points over time.
    T = y.shape[0]
    nper = 10
    for t in range(0, T, nper):
        c = t / float(T)
        c = [1 - c, 0., c]
        plt.scatter(y[t:t+nper, 0, 0], y[t:t+nper, 0, 1], color = c)

if __name__ == '__main__':

    if os.path.exists('ode_demo.pt'):
        print(device)
        save = torch.load('ode_demo.pt')
        func = ODEFunc().to(device)
        func.load_state_dict(save['func'])
        pred_y = odeint(func, true_y0, t)
#        loss = torch.mean(torch.abs(pred_y - true_y))
#        loss.backward()

        record = adjoint_calculate(t, pred_y, func, args.method, tol=1e-4)
        states = np.array([a[1].detach().numpy() for a in record])
        adjoints = np.array([a[2].detach().numpy() for a in record])

        w1_grad = np.array([a[3].detach().numpy() for a in record])
        b1_grad = np.array([a[4].detach().numpy() for a in record])
        w2_grad = np.array([a[5].detach().numpy() for a in record])
        b2_grad = np.array([a[6].detach().numpy() for a in record])

        w1_grad = w1_grad.reshape(w1_grad.shape[0], -1)
        w2_grad = w2_grad.reshape(w2_grad.shape[0], -1)

        print(states.shape, adjoints.shape, w1_grad.shape, b1_grad.shape, w2_grad.shape, b2_grad.shape)

        plt.figure(figsize=(8, 12))
        ax = plt.subplot(321)
        plot_flow(states, ax)
        plt.title('ODE Flow')
        ax = plt.subplot(322)
        plot_flow(adjoints, ax)
        plt.title('Adjoint Flow')

        ax = plt.subplot(323)
        plt.plot(t, w1_grad)
        plt.title('w1 Running Gradient')

        ax = plt.subplot(324)
        plt.plot(t, b1_grad)
        plt.title('b1 Running Gradient')

        ax = plt.subplot(325)
        plt.plot(t, w2_grad)
        plt.title('w2 Running Gradient')

        ax = plt.subplot(326)
        plt.plot(t, b2_grad)
        plt.title('b2 Running Gradient')

        plt.tight_layout()

        plt.figure(figsize=(8,4))
        plt.subplot(221)
        plt.plot(t, states[:, 0, 0])
        plt.title('$y_1$')
        plt.subplot(222)
        plt.plot(t, states[:, 0, 1])
        plt.title('$y_1$')

        plt.subplot(223)
        plt.plot(t, adjoints[:, 0, 0])
        plt.title('$a_1$')
        plt.subplot(224)
        plt.plot(t, adjoints[:, 0, 1])
        plt.title('$a_1$')

        plt.suptitle('Comparing Flows')
        plt.tight_layout()
        plt.show()

        exit()

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()

    save = {
        'optim': optimizer.state_dict(),
        'func': func.state_dict(),
        'loss': loss.item()
    }
    torch.save(save, 'ode_demo.pt')
