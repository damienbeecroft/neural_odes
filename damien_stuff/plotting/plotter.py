import torch
import matplotlib.pyplot as plt


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
    # # First plot
    # data1 = torch.load(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\cubic\output\data\backprop_cubic_nn2_1_100000_1000.pt")
    # loss_tracker = data1['loss_tracker']
    # epochs = list(data1['epochs'])
    # plt.plot(epochs,loss_tracker,label='Test Error')
    # plt.xlabel('Number of Training Iterations')
    # plt.ylabel('Average Loss')
    # plt.title('Error of Cubic Neural Net')
    # plt.legend()
    # plt.savefig(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\plotting\plots\backprop_cubic_nn2_1_100000_1000.png")
    # plt.show()

    # # Second plot 
    # data1 = torch.load(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\cubic\output\data\backprop_cubic_lin_3_5000_100.pt")
    # data2 = torch.load(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\cubic\output\data\adjoint_cubic_lin_3_5000_100.pt")
    # epochs = list(data1['epochs'])
    # loss_tracker = data1['loss_tracker']
    # loss_tracker2 = data2['loss_tracker']
    # time = data1['time_meter']
    # time2 = data2['time_meter']
    # plt.plot(epochs,loss_tracker,label='Backprop')
    # plt.plot(epochs,loss_tracker2,label='Adjoint')
    # plt.xlabel('Number of Training Iterations')
    # plt.ylabel('Average Loss')
    # plt.title('Error of Training Linear System')
    # plt.legend()
    # print(time.val)
    # print(time2.val)
    # plt.show()

    # Third Plot
    data1 = torch.load(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\cubic\output\data\adjoint_cubic_nn_3_5000_100.pt")
    loss_tracker = data1['loss_tracker']
    epochs = list(data1['epochs'])
    plt.plot(epochs,loss_tracker,label='Test Error')
    plt.xlabel('Number of Training Iterations')
    plt.ylabel('Average Loss')
    plt.title('Error of Cubic Neural Net')
    plt.legend()
    plt.savefig(r"C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\plotting\plots\adjoint_cubic_nn_3_5000_100.png")
    plt.show()

