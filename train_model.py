from datasets import PiecesDataset
from torch.utils.data import DataLoader
from model import ChessConvNet

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import os
import pickle
from time import time
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

def savetfigs(fig, axs,
              training_loss_vs_epoch, validation_loss_vs_epoch,
              training_acc_vs_epoch, validation_acc_vs_epoch):
    axs[0].clear()
    axs[1].clear()
    axs[0].plot(training_loss_vs_epoch, label='training')
    axs[0].plot(validation_loss_vs_epoch, label='validation')
    axs[0].legend(loc='upper right')
    axs[0].set_title('Loss vs epoch')
    axs[1].plot(training_acc_vs_epoch, label='training')
    axs[1].plot(validation_acc_vs_epoch, label='validation')
    axs[1].legend(loc='upper left')
    axs[1].set_title('Accuracy vs epoch')
    fig.savefig(os.getcwd() + '/figures/loss_curve_classifier.png', dpi=500)

def main():
    if not os.path.isdir('figures'):
        os.system('mkdir figures')
    if not os.path.isdir('parameters'):
        os.system('mkdir parameters')
    if torch.cuda.is_available():
        os.system('echo Running on GPU.\n')
    path_to_training_data = os.getcwd() + '/resources/training_dataset/pieces'
    path_to_validation_data = os.getcwd() + '/resources/validation_dataset/pieces'

    training_dataset = PiecesDataset(path_to_training_data)
    validation_dataset = PiecesDataset(path_to_validation_data)

    training_dataloader = DataLoader(training_dataset, batch_size=20, shuffle=True)  # need to shuffle since the files are orderd
    validation_dataloader = DataLoader(validation_dataset, batch_size=20)

    net = ChessConvNet()

    try:
        net.load_state_dict(torch.load('parameters/trained_model.pt'), strict=False)
        os.system('echo Parameters were loaded successfully!')
    except FileNotFoundError:
        pass

    loss_function = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.0001)

    if torch.cuda.is_available():
        net.cuda()

    if not os.path.isdir('lists'):
        os.system('mkdir lists')
    if os.path.isfile('lists/tlve.pickle'):
        os.system('echo Plotting lists were found in folder')
        with open('lists/tlve.pickle', 'rb') as fp:
            training_loss_vs_epoch = pickle.load(fp)
        with open('lists/tave.pickle', 'rb') as fp:
            training_acc_vs_epoch = pickle.load(fp)
        with open('lists/vave.pickle', 'rb') as fp:
            validation_acc_vs_epoch = pickle.load(fp)
        with open('lists/vlve.pickle', 'rb') as fp:
            validation_loss_vs_epoch = pickle.load(fp)
        os.system('echo Plotting lists were loaded successfully!')
    else:
        training_loss_vs_epoch = []
        validation_loss_vs_epoch = []
        training_acc_vs_epoch = []
        validation_acc_vs_epoch = []

    epochs = 50
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))


    for epoch in range(epochs):
        end_time = time()
        if epoch > 0:
            time_per_epoch = end_time - start_time
            os.system('echo Time for epoch %d: %.1f minutes' %(epoch - 1, time_per_epoch / 60))
            os.system('echo validation accuracy: %.2f' % validation_acc_vs_epoch[-1])
            os.system('echo training accuracy: %.2f' % training_acc_vs_epoch[-1])
            os.system('echo validation loss: %.2f' % validation_loss_vs_epoch[-1])
            os.system('echo training loss: %.2f' % training_loss_vs_epoch[-1])
            os.system('echo ================================================')
        start_time = time()
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

        for key, val in {'lists/tlve.pickle': training_loss_vs_epoch,
                         'lists/tave.pickle': training_acc_vs_epoch,
                         'lists/vave.pickle': validation_acc_vs_epoch,
                         'lists/vlve.pickle': validation_loss_vs_epoch}.items():
            with open(key, 'wb') as fp:
                pickle.dump(val, fp)

        if len(validation_loss_vs_epoch) > 1:
            if validation_loss_vs_epoch[-1] < min(validation_loss_vs_epoch[:-1]):
                torch.save(net.state_dict(), 'parameters/trained_model.pt')

        savetfigs(fig, axs,
                  training_loss_vs_epoch, validation_loss_vs_epoch,
                  training_acc_vs_epoch, validation_acc_vs_epoch)

if __name__ == '__main__':
    main()