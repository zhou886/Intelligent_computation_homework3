import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network import NetworkModuleBiLSTM
from dataset import MyDataset


def train():
    epoch = 300
    learning_rate = 0.001

    train_set = MyDataset(True)
    test_set = MyDataset(False)
    train_set_size = len(train_set)
    test_set_size = len(test_set)

    train_loader = DataLoader(train_set, batch_size=8,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=True)

    network = NetworkModuleBiLSTM()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        network = network.cuda()
        loss_function = loss_function.cuda()

    writer = SummaryWriter("logs")

    for i in range(epoch):
        print("epoch{0} starts.".format(i))
        total_train_loss = 0
        total_train_accuracy = 0
        total_test_loss = 0
        total_test_accuracy = 0

        for data in train_loader:
            s1, s2, label = data
            if torch.cuda.is_available():
                s1 = s1.cuda()
                s2 = s2.cuda()
                label = label.cuda()
            output = network(s1, s2)
            label = label.reshape((-1, 1))
            loss = loss_function(output.float(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(output, label)

            total_train_loss += loss.item()
            for i in range(len(label)):
                if abs(output[i] - label[i]) < 0.5:
                    total_train_accuracy += 1

        with torch.no_grad():
            for data in test_loader:
                s1, s2, label = data
                if torch.cuda.is_available():
                    s1 = s1.cuda()
                    s2 = s2.cuda()
                    label = label.cuda()
                output = network(s1, s2)
                label = label.reshape((-1, 1))
                loss = loss_function(output, label)

                total_test_loss += loss.item()
                for i in range(len(label)):
                    if abs(output[i] - label[i]) < 0.5:
                        total_test_accuracy += 1

        total_train_accuracy = 1.0*total_train_accuracy/train_set_size
        total_test_accuracy = 1.0*total_test_accuracy/test_set_size

        writer.add_scalar('train_loss', total_train_loss, i)
        writer.add_scalar('train_accuracy', total_train_accuracy, i)
        writer.add_scalar('test_loss', total_test_loss, i)
        writer.add_scalar('test_accuracy', total_test_accuracy, i)


if __name__ == '__main__':
    train()
