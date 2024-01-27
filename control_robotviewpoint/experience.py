import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import serial

'''ser = serial.Serial('COM3', 9600)'''

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

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.layer6 = self.conv_module(256, 256, 0.5)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


hyper_param_epoch = 20
hyper_param_batch = 8
hyper_param_learning_rate = 0.001


transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.589, 0.209, 0.411], std=[0.290, 0.153, 0.146])
                                      ])



test_data_set = CustomImageDataset(data_set_path=r"C:\Users\tpals\OneDrive\Desktop\datattest", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = test_data_set.num_classes
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
                  .format(e + 1, hyper_param_epoch, loss.item()))'''

# Test the model
sound_list = []
model = torch.load(r"C:\Users\tpals\model\model_final_1031.pt", map_location=device)
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

        # 이미지 파일 경로에서 파일 이름 추출 및 확장자 제거
        image_path = test_data_set.image_files_path[total - 1]
        image_name, _ = os.path.splitext(os.path.basename(image_path))
        sound_list.append(predicted.item())
        print('이미지 파일 이름:', image_name)
        print('예측 라벨:', predicted.item())

    #print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
print(sound_list)
if 5 in sound_list:
    target = sound_list.index(5)
    print(target)
else:
    target = sound_list.index(min(sound_list))
    print(target)

print('로봇의 목표 방향:', target+1)
data = target+1
data = str(data)
data = data.encode()
'''ser.write(data)
ser.close()
'''