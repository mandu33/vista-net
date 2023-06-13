import cv2 as cv
import numpy as np
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision import transforms
from einops import rearrange, reduce, repeat
from tqdm import tqdm
from torchinfo import summary
import json

#调用sklearn计算指标
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from datetime import datetime
#获取验证图片数据集的元信息
def get_valid_meta_data():
    data = []
    with open('dataset/valid.txt', 'r') as f:
        file_content:str = f.read()
        data_content = file_content.strip().split("\n")[1:]
        data.extend([d.split(',') for d in data_content])
    for d in data:
        d[0] = d[0].split("'")[1]
    return data
#获取训练图片数据集的元信息
def get_train_meta_data():
    data = []
    with open('dataset/train.txt', 'r') as f:
        file_content:str = f.read()
        data_content = file_content.strip().split("\n")[1:]
        data.extend([d.split(',') for d in data_content])
    for d in data:
        d[0] = d[0].split("'")[1]
    return data
#获取测试图片数据集的元信息
def get_test_meta_data():
    data = []
    with open('dataset/test.txt', 'r') as f:
        file_content:str = f.read()
        data_content = file_content.strip().split("\n")[1:]
        data.extend([d.split(',') for d in data_content])
    for d in data:
        d[0] = d[0].split("'")[1]
    return data
#获取训练图片数据集
def get_train_data_item(meta_item):
    file_path = f"dataset/train_lower/{meta_item[0].lower()}/{meta_item[1].lower()}"
    score = int(meta_item[2]) if meta_item[2]!='None' else 0
    image_data = cv.imread(file_path)
    return image_data, score
#获取验证图片数据集
def get_valid_data_item(meta_item):
    file_path = f"dataset/valid_lower/{meta_item[0].lower()}/{meta_item[1].lower()}"
    score = int(meta_item[2]) if meta_item[2]!='None' else 0
    image_data = cv.imread(file_path)
    return image_data, score
#获取测试图片数据集
def get_test_data_item(meta_item):
    file_path = f"dataset/test_lower/{meta_item[0].lower()}/{meta_item[1].lower()}"
    score = int(meta_item[2]) if meta_item[2]!='None' else 0
    image_data = cv.imread(file_path)
    return image_data, score
#验证集数据导入
class ValidDataset(Dataset):
    def __init__(self):
        self.meta_data = get_valid_meta_data()
        self.data = {}
        self.target = {}
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
        ])
    def __len__(self):
        return len(self.meta_data[:])
    def __getitem__(self, idx):
        image = self.data.get(idx)
        label = self.target.get(idx)
        if image == None:
            img, label = get_valid_data_item(self.meta_data[idx])
            image = self.preprocess(img)
            self.data[idx] = image
            self.target[idx] = label
        return image, label
#测试集数据导入
class TestDataset(Dataset):
    def __init__(self):
        self.meta_data = get_test_meta_data()
        self.data = {}
        self.target = {}
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
        ])

    def __len__(self):
        return len(self.meta_data[:])

    def __getitem__(self, idx):
        image = self.data.get(idx)
        label = self.target.get(idx)
        if image == None:
            img, label = get_test_data_item(self.meta_data[idx])
            image = self.preprocess(img)
            self.data[idx] = image
            self.target[idx] = label
        return image, label
#训练数据集导入
class MyDataset(Dataset):
    def __init__(self):
        self.meta_data = get_train_meta_data()
        self.data = {}
        self.target = {}
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
        ])

    def __len__(self):
        return len(self.meta_data[:])

    def __getitem__(self, idx):
        image = self.data.get(idx)
        label = self.target.get(idx)
        if image == None:
            img, label = get_train_data_item(self.meta_data[idx])
            image = self.preprocess(img)
            self.data[idx] = image
            self.target[idx] = label
        return image, label

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, conv_size):
        super().__init__()
        self.layer0 = nn.Conv2d(c_in, c_out, conv_size, stride=(1,1), padding="same")
        self.layer1 = nn.Conv2d(c_in, c_out, conv_size, stride=(1,1), padding="same")
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x + self.layer1(self.relu(self.layer0(x)))
        x = self.relu(x)
        return x

class ResBlock_down(nn.Module):
    def __init__(self, c_in, c_out, conv_size):
        super().__init__()
        self.layer0 = nn.Conv2d(c_in, c_out, conv_size, stride=(2,2), padding=(1,1))
        self.layer1 = nn.Conv2d(c_out, c_out, conv_size, stride=(1,1), padding=(1,1))
        self.layer2 = nn.Conv2d(c_in, c_out, conv_size, stride=(2,2), padding=(1,1))
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.layer2(x) + self.layer1(self.relu(self.layer0(x)))
        x = self.relu(x)
        return x

#resnet18网络
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.conv2 = nn.Sequential(
            ResBlock(64, 64, 3),
            ResBlock(64, 64, 3),
            ResBlock_down(64, 128, 3),
            ResBlock(128, 128, 3),
            ResBlock_down(128, 256, 3),
            ResBlock(256, 256, 3),
            ResBlock_down(256, 512, 3),
            ResBlock(512, 512, 3),
        )
        self.pool = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(512, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.softmax(x)
        y = self.layer(x)

        return y

def report(lables,prediction):
  measure_result = classification_report(lables, prediction)#输出模型评估报告
  print("------report------")
  print(measure_result)
  with open("./loss/resnet/report.txt", 'w') as file1:
          file1.write(str(measure_result))
#   #包含了五种数据集的各个指标报告，需要进行进一步的平均处理
#   #precison
  precision = precision_score(lables, prediction, average="micro")
# #   with open("./loss/precision_test.txt", 'w') as file1:
# #           file1.write(str(precision_score_average_None))
#   #recall
  recall = recall_score(lables, prediction, average="micro")
# #   with open("./loss/recall_test.txt", 'w') as file2:
# #           file2.write(str(recall_score_average_None))
#   #F1 score
  f1 = f1_score(lables, prediction, average="micro")
# #   with open("./loss/F1_test.txt", 'w') as file3:
# #           file3.write(str(f1_score_average_None))
  return precision, recall, f1

    
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 20
    batch_size = 32
    #导入数据集
    training_data = MyDataset()
    #dataloader加载数据集
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    #验证集
    valid_data = ValidDataset()
    valid_dataloader = DataLoader(valid_data,batch_size=batch_size)
    #测试集
    test_data = TestDataset()
    test_dataloader = DataLoader(test_data,batch_size=batch_size)

    tlen = test_data.__len__()
    model = ResNet18()
    #打印网络结构
    summary(model, input_size=(batch_size, 3, 224, 224))
    #交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    #adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # meta_data = get_train_meta_data()
    # img, score = get_train_data_item(meta_data[0])
    # print(img.shape, score)
    # img_data = torch.tensor([preprocess(img).tolist()])
    # print("img_data", img_data.shape)
    # outimg = rearrange(img_data[0], "c h w -> h w c")
    # print("outimg", outimg.shape)
    # cv.imwrite("tmp.png", outimg.numpy()*255)
    # y = torch.tensor([score], dtype=torch.int64)
    # model.eval()
    # y_pred = model(img_data)
    # print("y_pred", y_pred)
    # print("y", y)
    # loss = loss_fn(y_pred, y)
    
    model.to(device)
    model.train()
    print("-----开始训练resnet18模型-----")
    train_loss = []#记录train_loss
    valid_loss = []#记录valid_loss
    for e in (range(epochs)):
        for batch, (x, y) in (enumerate(tqdm(train_dataloader))):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.tolist())
        for batch, (vx, vy) in (enumerate(tqdm(valid_dataloader))):
            vx,xy = vx.to(device), vy.to(device)
            vpred = model(vx)
            vloss = loss_fn(vpred.cpu(), vy.cpu())
            valid_loss.append(vloss.tolist())
    print("-----completed valid-----")
    #print(train_loss)
    #print(valid_loss)
    with open("./loss/resnet/train_resnet_loss.txt", 'w') as train_loss_file:
          train_loss_file.write(str(train_loss))
    with open("./loss/resnet/valid_resnet_loss.txt", 'w') as valid_loss_file:
          valid_loss_file.write(str(valid_loss))
    print("{} training and validing Finished!".format(datetime.now()))

    #记录测试集各项指标
    test_loss  = 0
    tlabel = []
    tprediction = []
    print("-----start test-----")
    accuracy = 0
    error = 0
    precision = 0
    recall = 0
    f1 = 0
    with torch.no_grad():
        #开始测试
        for batch, (tx, ty) in (enumerate(tqdm(test_dataloader))):
            tx, ty = tx.to(device), ty.to(device)
            pred = model(tx)
            #print(pred)
            loss = loss_fn(pred, ty)
            test_loss += loss

            tlabel.append(ty.cpu().numpy().tolist())
            #pred = torch.argmax(pred, dim=1) # di表示在哪维度（列）上哪个值最大
            max_value, max_index = torch.max(pred, dim=1)

            tprediction.append(max_index.cpu().numpy().tolist())

            label_list = [item for sublist in tlabel for item in sublist]
            prediction_list = [item for sublist in tprediction for item in sublist]

            sklearn_accuracy = accuracy_score(label_list, prediction_list) 
            sklearn_error = 1.0 - sklearn_accuracy
            sklearn_precision = precision_score(label_list, prediction_list, average='micro')
            sklearn_recall = recall_score(label_list, prediction_list, average='micro')
            sklearn_f1 = f1_score(label_list, prediction_list, average='micro')
            
            accuracy += sklearn_accuracy;
            error += sklearn_error
            precision += sklearn_precision
            recall += sklearn_recall
            f1 += sklearn_f1

    test_loss /= test_data.__len__()
    print("test loss is "+str(test_loss))
    print("--------------------")
    
    result = {'loss':test_loss,'accuracy':accuracy*batch_size/tlen,'error':error*batch_size/tlen,'precision':precision*batch_size/tlen,'recall':recall*batch_size/tlen,'f1':f1*batch_size/tlen}

    with open('./loss/resnet/test_result.txt', 'w') as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

    print("-----completed test-----")

if __name__ == "__main__":
    main()