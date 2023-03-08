import torch
import torchvision
import lenet5_moudel

batch_size = 64
# 导入训练集和测试集
train_data = torchvision.datasets.MNIST('../mnist_data', train=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()]),
                                        download=False)
test_data = torchvision.datasets.MNIST('../mnist_data', train=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()]),
                                       download=False)

# train_data = torchvision.datasets.CIFAR10('../cifar', train=True, transform=torchvision.transforms.ToTensor(),
# download=False)
# test_data = torchvision.datasets.CIFAR10('../cifar', train=False,
# transform=torchvision.transforms.ToTensor(),download=False)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 检查数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

# 设置训练模式
device = torch.device('cuda:0')
# 设置损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 导入网络模型
lenet5 = lenet5_moudel.Lenet().to(device)
# 设置优化器
optimizer = torch.optim.SGD(lenet5.parameters(), lr=0.01)

# 设置网络训练参数
epoch = 20
total_train_step = 0

# 开始训练
lenet5.train()
for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i + 1))
    for data, label in train_dataloader:
        data, label = data.to(device), label.to(device)
        outputs = lenet5(data)
        loss = loss_fn(outputs, label)
        # 反向传播+参数优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))

    # 测试数据
    total_test_step = 0
    total_test_loss = 0
    total_test_acc = 0
    lenet5.eval()
    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.to(device), label.to(device)
            outputs = lenet5(data)
            loss = loss_fn(outputs, label)
            total_test_step = total_test_step + 1
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == label).sum()
            total_test_acc = total_test_acc + accuracy

    print("整体测试集上的loss:{}".format(total_test_loss / total_test_step))
    print("整体测试集上的acc:{}".format(total_test_acc / test_data_size))
