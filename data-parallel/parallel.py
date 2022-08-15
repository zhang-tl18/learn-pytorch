from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 构建一个假的数据集
class RandomDataset(Dataset):
    def __init__(self, size, length) -> None:
        super().__init__()
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                        batch_size=batch_size, shuffle=True)

# 网络模型
class Model(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        return output

# 创建模型
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.cuda()

# 运行模型
for data in rand_loader:
    input = data.cuda()
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
    