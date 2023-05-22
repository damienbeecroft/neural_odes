import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # num_its = 5000
    # test_freq = 100
    data = torch.load("C:\Users\damie\OneDrive\UW\amath575\project\neural_odes\damien_stuff\cubic\output\data\cubic_nn_5000_100.pt")
    loss_tracker = data['loss_tracker']
    epochs = list(data['epochs'])
    # plt.plot(list(range(0,num_its,test_freq)),loss_tracker)
    plt.xlabel('Number of Training Iterations')
    plt.ylabel('Average Loss')
    plt.show()


    # plt.plot(list(range(0,num_its,test_freq)),loss_tracker)
    # plt.xlabel('Number of Training Iterations')
    # plt.ylabel('Average Loss')
    # plt.show()