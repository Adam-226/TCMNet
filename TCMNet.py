import os
import time  # 导入time模块
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# 检查MPS是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')

# 设置数据路径
data_dir = '/Users/Adam/PycharmProjects/AI_lab7/cnn图片'

# 数据增强和预处理
# 定义一个图像预处理的变换管道（transform pipeline）
# 定义了一系列顺序执行的图像变换操作，这些操作通过 transforms.Compose 进行组合，按顺序依次应用于输入图像。
# 这些操作依次为：
# 1. 将输入图像调整为大小为 128x128 像素。无论输入图像的原始大小是多少，它都会被缩放到这个固定尺寸。
# 2. 将图像从 PIL 图像或 numpy 数组转换为 PyTorch 的张量（tensor）。同时，它会将图像的像素值从范围 [0, 255] 缩放到 [0, 1]（如果图像是 RGB 的，每个通道独立处理）。
# 3. 对张量进行标准化处理。标准化是通过减去均值并除以标准差来完成的。这里提供的 mean 和 std 是针对 ImageNet 数据集的预训练模型常用的均值和标准差值：
#    mean=[0.485, 0.456, 0.406] 对应于三个通道（RGB）的均值。
#    std=[0.229, 0.224, 0.225] 对应于三个通道（RGB）的标准差。

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练数据集
# ImageFolder用于加载以文件夹组织的图像数据集。假设每个子文件夹的名称是一个类别名，子文件夹中的图像都是属于该类别的。
# 使用 os.path.join 函数来构建训练数据所在目录的路径。
# DataLoader 用于将数据集加载为小批量数据，以便进行批量训练。它支持多线程数据加载，自动化的数据洗牌，以及批量数据的自动化处理。
# batch_size=32 指定了每个小批量数据的大小，即每次从数据集中加载 32 张图像进行训练。
# shuffle=True 指定在每个训练周期开始时对数据进行随机打乱，确保模型不会记住数据的顺序，有助于提高模型的泛化能力。

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义自定义数据集类加载测试数据
class TestDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list   # file_list: 一个包含图像文件路径的列表。
        self.transform = transform

    def __len__(self):
        return len(self.file_list)   # 返回列表元素的数目

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')    # 使用 PIL 打开图像并转换为 RGB 模式。
        if self.transform:
            image = self.transform(image)    # 对图像应用变换。
        label = 0  # 测试集不需要标签，所以这里可以随便给一个
        return image, label


# 获取测试文件路径
test_dir = os.path.join(data_dir, 'test')    # 定义测试数据目录
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]   # 最终得到一个列表 test_files，其中包含所有测试图像文件（以 .jpg 结尾）的完整路径。
test_dataset = TestDataset(test_files, transform=transform)   # 创建自定义测试数据集
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)    # 创建数据加载器，指定每个小批量数据的大小为32，指定不对数据进行随机打乱。


# 定义卷积神经网络架构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()    # 调用父类 nn.Module 的初始化方法。
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)     # 第一个卷积层，输入通道数为 3（RGB 图像），输出通道数为 32，卷积核大小为 3x3，padding 为 1（保持输入尺寸）。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)    # 第二个卷积层，输入通道数为 32，输出通道数为 64，卷积核大小为 3x3，padding 为 1。
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)   # 第三个卷积层，输入通道数为 64，输出通道数为 128，卷积核大小为 3x3，padding 为 1。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # 最大池化层，池化窗口大小为 2x2，步幅为 2。
        self.fc1 = nn.Linear(128 * 16 * 16, 512)   # 第一个全连接层，输入特征数为 128 * 16 * 16（经过卷积和池化后的图像特征展平），输出特征数为 512。
        self.fc2 = nn.Linear(512, 5)   # 第二个全连接层，输入特征数为 512，输出特征数为 5（假设有 5 个类别进行分类）。
        self.dropout = nn.Dropout(0.5)    # Dropout 层，用于防止过拟合，随机将 50% 的神经元设置为 0。

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 输入 x 经过第一个卷积层 conv1，然后经过 ReLU 激活函数，再经过最大池化层 pool。
        x = self.pool(torch.relu(self.conv2(x)))   # 输入 x 经过第二个卷积层 conv2，然后经过 ReLU 激活函数，再经过最大池化层 pool。
        x = self.pool(torch.relu(self.conv3(x)))   # 输入 x 经过第三个卷积层 conv3，然后经过 ReLU 激活函数，再经过最大池化层 pool。
        x = x.view(-1, 128 * 16 * 16)    # 将经过卷积和池化后的特征图展平为一维张量，作为全连接层的输入。
        x = torch.relu(self.fc1(x))    # 输入 x 经过第一个全连接层 fc1，然后经过 ReLU 激活函数。
        x = self.dropout(x)   # 输入 x 经过 Dropout 层。
        x = self.fc2(x)   # 输入 x 经过第二个全连接层 fc2。
        return x


model = CNN().to(device)

# 设置损失函数和优化器
# CrossEntropyLoss 在分类任务中用来计算模型预测的类别分布与真实标签之间的差异。它接受未归一化的预测（logits）和真实标签（类别索引），然后计算损失。
# Adam 优化器结合了 AdaGrad 和 RMSProp 的优点，能够在训练过程中自动调整学习率。
# model.parameters(): 传递给优化器的参数是模型的所有可训练参数。model 是已经定义好的神经网络模型实例。
# lr=0.001: 设置优化器的学习率（learning rate）。学习率是一个超参数，决定了每次参数更新的步长。
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []
train_accuracies = []
test_accuracies = []

# 训练模型
for epoch in range(num_epochs):
    start_time = time.time()  # 记录开始时间
    model.train()   # 将模型设置为训练模式
    # 初始化损失、正确预测数和总样本数
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 清除优化器中之间积累的梯度
        outputs = model(inputs)   # 进行前向传播，得到模型的输出。
        loss = criterion(outputs, labels)  # 计算模型输出和真实标签之间的损失。
        loss.backward()   # 执行反向传播，计算模型参数的梯度。
        optimizer.step()   # 使用优化器更新模型参数。

        # 更新训练损失和准确率
        running_loss += loss.item()   # 累积每个批次的损失。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)    # 统计总样本数。
        correct += (predicted == labels).sum().item()   # 统计正确预测的样本数。

    # 计算并存储每个周期的平均训练损失和准确率
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 记录周期的结束时间并计算运行时间
    end_time = time.time()  # 记录结束时间
    epoch_duration = end_time - start_time  # 计算运行时间

    # 打印每个周期的训练信息
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Time: {epoch_duration:.2f}s')

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # 根据文件名判断标签，假设文件名格式是 "baihe01.jpg", "dangshen01.jpg", ...
            labels = [train_dataset.class_to_idx[os.path.basename(f).rsplit('0', 1)[0]] for f in test_files]
            labels = torch.tensor(labels).to(device)
            total += len(labels)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# 画出训练过程的loss曲线和准确率曲线
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Training accuracy')
plt.plot(epochs, test_accuracies, 'ro-', label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Testing Accuracy')

plt.show()
