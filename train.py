import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from network import NetworkModuleBiLSTM, NetworkModuleTransformer, NetworkModuleBERT
from dataset import MyDataset
import os


def train(module: str = "BiLSTM", log_dir: str = "./logs", module_save_dir: str = "./modules", epoch: int = 100, learning_rate: float = 0.005, batchsize: int = 8):
    train_set = MyDataset(True)
    test_set = MyDataset(False)
    train_set_size = len(train_set)
    test_set_size = len(test_set)

    train_loader = DataLoader(train_set, batch_size=batchsize,
                              shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batchsize,
                             shuffle=True)

    if module == "BiLSTM":
        network = NetworkModuleBiLSTM()
    elif module == "Transformer":
        network = NetworkModuleTransformer()
    elif module == "BERT":
        network == NetworkModuleBERT()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        network = network.cuda()
        loss_function = loss_function.cuda()

    writer = SummaryWriter(log_dir)
    if not os.path.exists(module_save_dir):
        os.mkdir(module_save_dir)

    print("Network Module:", module)
    step = 1
    for i in range(epoch):
        print("epoch{0} start".format(i))
        total_train_loss = 0
        total_train_accuracy = 0
        total_test_loss = 0
        total_test_accuracy = 0

        for data in train_loader:
            s1, s2, label, len1, len2 = data
            if torch.cuda.is_available():
                s1 = s1.cuda()
                s2 = s2.cuda()
                label = label.cuda()
            output = network(s1, s2, len1, len2)
            label = label.reshape((-1, 1))
            loss = loss_function(output.float(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            for i in range(len(label)):
                if abs(output[i] - label[i]) < 0.5:
                    total_train_accuracy += 1

        with torch.no_grad():
            for data in test_loader:
                s1, s2, label, len1, len2 = data
                if torch.cuda.is_available():
                    s1 = s1.cuda()
                    s2 = s2.cuda()
                    label = label.cuda()
                output = network(s1, s2, len1, len2)
                label = label.reshape((-1, 1))
                loss = loss_function(output, label)

                total_test_loss += loss.item()
                for i in range(len(label)):
                    if abs(output[i] - label[i]) < 0.5:
                        total_test_accuracy += 1

        total_train_accuracy = 1.0*total_train_accuracy/train_set_size
        total_test_accuracy = 1.0*total_test_accuracy/test_set_size

        writer.add_scalar('train_loss', total_train_loss, step)
        writer.add_scalar('train_accuracy', total_train_accuracy, step)
        writer.add_scalar('test_loss', total_test_loss, step)
        writer.add_scalar('test_accuracy', total_test_accuracy, step)
        if step % 5 == 0:
            torch.save(network.state_dict(), os.path.join(
                module_save_dir, "epoch{}_lr{}_batchsize{}_accuracy{}.pth".format(step, learning_rate, batchsize, round(total_test_accuracy, 2))))
        step += 1

    writer.close()


if __name__ == '__main__':
    for i in range(5):
        train("BiLSTM", "./BiLSTM_logs_{0}".format(i+1),
              "./BiLSTM_Modules_{0}".format(i+1))
        train("Transformer",
              "./Transformer_logs_{0}".format(i+1), "./Transformer_Modules_{0}".format(i+1))
