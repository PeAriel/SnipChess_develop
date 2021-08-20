from datasets import PiecesDataset
from torch.utils.data import DataLoader
from model import ChessConvNet

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_accuracy_and_loss(loss_func, dataloader, net):
    correct = 0
    loss = 0
    total = 0
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    with torch.no_grad():
        for nbatch, data_and_targets in enumerate(dataloader, 1):
            data, target = data_and_targets
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            prediction = net(data)
            loss += loss_func(prediction, target).item()
            pred = torch.argmax(prediction, dim=1)
            correct += len(torch.where(pred == target)[0])
            total += len(target)

    accuracy = correct / total
    loss /= nbatch

    return accuracy, loss

def main():
    if torch.cuda.is_available():
        os.system('echo Running on GPU.\n')
    path_to_training_data = os.getcwd() + '/resources/training_dataset/pieces'
    path_to_validation_data = os.getcwd() + '/resources/validation_dataset/pieces'

    training_dataset = PiecesDataset(path_to_training_data)
    validation_dataset = PiecesDataset(path_to_validation_data)

    training_dataloader = DataLoader(training_dataset, batch_size=20, shuffle=True)  # need to shuffle since the files are orderd
    validation_dataloader = DataLoader(validation_dataset, batch_size=20)

    net = ChessConvNet()
    loss_function = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001)

    if torch.cuda.is_available():
        net.cuda()

    n_epochs = 50
    training_loss_vs_epoch = []
    validation_loss_vs_epoch = []
    training_acc_vs_epoch = []
    validation_acc_vs_epoch = []
    pbar = tqdm(range(n_epochs))

    for epoch in pbar:
        if len(validation_loss_vs_epoch) > 1:
            pbar.set_description('val acc:' + '{0:.2f}'.format(validation_acc_vs_epoch[-1]) +
                                 ', train acc:' + '{0:.2f}'.format(training_acc_vs_epoch[-1]) +
                                ', val loss:' + '{0:.2f}'.format(validation_loss_vs_epoch[-1]) +
                                ', train loss:' + '{0:.2f}'.format(training_loss_vs_epoch[-1]))

        net.train()
        for data, target in training_dataloader:
            optimizer.zero_grad()
            prediction = net(data)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()

        net.eval()
        training_accuracy, training_loss = compute_accuracy_and_loss(loss_function,training_dataloader, net)
        validation_accuracy, validation_loss = compute_accuracy_and_loss(loss_function,validation_dataloader, net)

        training_loss_vs_epoch.append(training_loss)
        training_acc_vs_epoch.append(training_accuracy)
        validation_acc_vs_epoch.append(validation_accuracy)
        validation_loss_vs_epoch.append(validation_loss)

        if len(validation_loss_vs_epoch) > 1:
            if validation_loss_vs_epoch[-1] < min(validation_loss_vs_epoch[:-1]):
                torch.save(net.state_dict(), 'trained_model.pt')

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].plot(training_loss_vs_epoch, label='training')
    ax[0].plot(validation_loss_vs_epoch, label='validation')
    ax[0].legend(loc='upper right')
    ax[0].set_title('Loss vs epoch')
    ax[1].plot(training_acc_vs_epoch, label='training')
    ax[1].plot(validation_acc_vs_epoch, label='validation')
    ax[1].legend(loc='upper left')
    ax[1].set_title('Accuracy vs epoch')

    plt.savefig(os.getcwd() + '/loss_curve_classifier.png', dpi=500)
    plt.show()

if __name__ == '__main__':
    main()