import os
from PIL import Image

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms


class CustomImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length


class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16, 0.5)
        self.layer2 = self.conv_module(16, 32, 0.5)
        self.layer3 = self.conv_module(32, 64, 0.5)
        self.layer4 = self.conv_module(64, 128, 0.5)
        self.layer5 = self.conv_module(128, 256, 0.5)
        self.layer6 = self.conv_module(256, 256, 0.5)
        #self.layer7 = self.conv_module(512, 256, 0.5)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        #out = self.layer7(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)

        return out

    def conv_module(self, in_num, out_num, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout_prob)  # 드롭아웃 레이어 추가
        )

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


hyper_param_epoch = 40
hyper_param_batch = 64
hyper_param_learning_rate = 0.005

transforms_train = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.589, 0.209, 0.411], std=[0.290, 0.153, 0.146])])

transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.589, 0.209, 0.411], std=[0.290, 0.153, 0.146])])

train_data_set = CustomImageDataset(data_set_path=r"C:\Users\tpals\OneDrive\Desktop\trainset", transforms=transforms_train)
print(len(train_data_set))
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path=r"C:\Users\tpals\OneDrive\Desktop\testset", transforms=transforms_test)
print(len(test_data_set))
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = train_data_set.num_classes
criterion = nn.CrossEntropyLoss()
custom_model = CustomConvNet(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
print(custom_model)

# 모델의 각 레이어 및 파라미터 접근
for name, param in custom_model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()}")

# 첫 번째 합성곱 레이어의 가중치 시각화 (예시)
if hasattr(custom_model, 'layer1') and hasattr(custom_model.layer1, '0'):
    first_conv_layer = custom_model.layer1[0]
    filters = first_conv_layer.weight.data.cpu()
    num_filters = filters.size(0)

# 필터를 그리드로 표시
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_filters):
        row = i // 4
        col = i % 4
        ax = axs[row, col]
        filter_i = filters[i].numpy()
        ax.imshow(filter_i[0], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i + 1}')
    plt.show()
else:
    print("The model does not have the specified layer for visualization.")
'''for e in range(hyper_param_epoch):
    for i_batch, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)

        # Forward pass
        outputs = custom_model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i_batch + 1) % hyper_param_batch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(e + 1, hyper_param_epoch, loss.item()))

# Test the model

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for item in test_loader:
        images = item['image'].to(device)
        labels = item['label'].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        print('예측 = ', predicted, '실제값 = ', labels)

    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))'''

train_losses = []
test_losses = []
test_accuracies = []

for e in range(hyper_param_epoch):
    running_loss = 0.0

    for i_batch, item in enumerate(train_loader):

        images = item['image'].to(device)
        labels = item['label'].to(device)

        # Forward pass
        outputs = custom_model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i_batch + 1) % hyper_param_batch == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(e + 1, hyper_param_epoch, loss.item()))
        train_losses.append(loss.item())
        running_loss = running_loss + loss.item()
    # 에폭마다 평균 손실 계산
    train_loss = running_loss / len(train_loader)
    train_losses.append(loss.item())

    # 테스트 데이터셋에서 모델 평가 및 결과 기록
    custom_model.eval()
    #model = torch.load(r"C:\Users\tpals\model\model_0925_2.pt")
    #model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for item in test_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = custom_model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()
            test_loss = test_loss + loss.item()
    # 에폭마다 평균 테스트 손실 및 정확도 기록
    test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)

    # 에폭마다 손실과 정확도를 출력합니다.
    print(f'Epoch [{e + 1}/{hyper_param_epoch}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Test Loss: {test_loss:.4f}, '
          f'Test Accuracy: {accuracy:.2f}%')

torch.save(custom_model, r"C:\Users\tpals\model\model_final_test")
output_path = r"C:\Users\tpals\model\graph"
# 손실 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', marker='o')
#plt.plot(test_losses, label='Test Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'train_loss_graph.png'), bbox_inches='tight', pad_inches=0)  # 'loss_graph.png' 파일로 저장
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(test_losses, label='Test Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'test_loss_graph.png'), bbox_inches='tight', pad_inches=0)  # 'loss_graph.png' 파일로 저장
plt.show()

# 정확도 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(test_accuracies, label='Test Accuracy', marker='o', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'accuracy_graph.png'), bbox_inches='tight', pad_inches=0)  # 'accuracy_graph.png' 파일로 저장
plt.show()
