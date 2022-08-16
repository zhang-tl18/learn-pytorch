import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt

n_epochs = 3
batch_size_trian = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)


# 加载数据
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]
)

train_set = torchvision.datasets.MNIST(root='./data/', train=True, 
                                download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size_trian, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

def show_some_imgs():
    examples = iter(test_loader)
    example_data, example_targets = examples.next()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

if __name__ == '__main__':
    # 初始化网络和优化器
    network = Net()
    if torch.cuda.is_available():
        network = nn.DataParallel(network)
        network.cuda()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # 训练模型
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []
    
    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), loss.item()
                ))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(network.state_dict(), './model.pth')     # .load_state_dict(state_dict)
                torch.save(optimizer.state_dict(), './optimizer.pth')

    # 测试模型
    def test(epoch):
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = network(data)
                test_loss += F.nll_loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            test_counter.append(epoch * len(train_loader.dataset))
            print(
                '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)
                ))
    
    start_beginning = time.time()
    test(0)
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train(epoch)
        print('Train Epoch: {}\tCost time: {}'.format(epoch, time.time()-start))
        test(epoch)
    print('\nFinished Training! Total cost time: {}\n'.format(time.time()-start_beginning))

    # 绘制训练曲线
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
