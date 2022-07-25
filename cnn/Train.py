from torch import optim as optim
from torch import utils as utils
from torch import nn as nn
from torch import no_grad as no_grad
from torch import device as device
from torch import unsqueeze as unsqueeze

from torch import max as max
from torch import save as model_save
from torch import load as model_load
from MyNet import MyNet as MyNet

from torchvision import transforms as T
from torchvision import datasets as ExtData

import numpy as np
import os
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt

# グラフのスタイルを指定
plt.style.use('seaborn-darkgrid')


def train_epoch(model, optimizer, criterion, train_loader, device):
    train_loss = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def inference(model, criterion, test_loader, device):
    model.eval()
    test_loss=0

    with no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
        test_loss = test_loss / len(test_loader.dataset)
    return test_loss


def run(num_epochs, model, optimizer, criterion, train_loader, test_loader, device):
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        test_loss = inference(model, criterion, test_loader, device)

        print(f'Epoch [{epoch+1}], train_Loss : {train_loss:.4f}, test_Loss : {test_loss:.4f}')
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
    return train_loss_list, test_loss_list, model


def show(num_epochs, train_loss_list, test_loss_list):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(range(num_epochs), train_loss_list, c='b', label='train loss')
    ax.plot(range(num_epochs), test_loss_list, c='r', label='test loss')
    ax.set_xlabel('epoch', fontsize='20')
    ax.set_ylabel('loss', fontsize='20')
    ax.set_title('training and test loss', fontsize='20')
    ax.grid()
    ax.legend(fontsize='25')
    plt.show()


def predict(model, current_dir, test_dataset):
    #画像の取り込み
    sample_image = Image.open(f"{current_dir}/sample_image.jpeg")
    sample_image = sample_image.resize((32, 32))
    sample_image_tensor = T.functional.to_tensor(sample_image)
    
    # テストデータの分類クラス
    classes = test_dataset.classes

    # テストデータから画像一枚分取り出す
    # sample_image = test_dataset[0][0]
    # T.functional.to_pil_image(sample_image).show()
    
    # pytorch はバッチサイズ毎の処理を前提としているため, 4次元へ変換する
    sample_image_tensor = unsqueeze(sample_image_tensor, 0)
    sample_image_tensor = sample_image_tensor.to(device("cpu"))
    pred = model(sample_image_tensor)
    _, pred_pos = max(pred, 1)
    
    # 分類ラベルのうちどれが一番確率が高いかを出力
    sample_image.show()
    print(f"predicted: {classes[pred_pos]}")



def main():
    # current dir
    current_dir = Path(__file__).resolve().parent

    # CNN
    model = MyNet()
    # 正規化
    normalize = T.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))  
    # Tensor化
    to_tensor = T.ToTensor()
    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()
    # 最適化手法を設定
    optimizer = optim.Adam(model.parameters())
    # epoch 数
    num_epochs = 30

    transform_train = T.Compose([to_tensor, normalize])
    transform_test = T.Compose([to_tensor, normalize])
    
    train_dataset = ExtData.CIFAR10(f"{current_dir}/data", train=True, download=True, transform=transform_train)
    test_dataset = ExtData.CIFAR10(f"{current_dir}/data", train=False, download=True, transform=transform_test)

    batch_size = 64
    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    is_model = os.path.isfile(f"{current_dir}/cnn_model.pth")    
    if not is_model:
        print("model data does not exists...")
        # Run
        train_loss_list, test_loss_list, model = run(num_epochs, model, optimizer, criterion, train_loader, test_loader, device("cpu"))
        # save trained model
        model_save(model.state_dict(), f"{current_dir}/cnn_model.pth")
        # FigureOut
        show(num_epochs, train_loss_list, test_loss_list)

    else:
        print("model data exists (load data)...")
        model.load_state_dict(model_load(f"{current_dir}/cnn_model.pth"))
        model.eval()

    # if you set sample_image.jpeg in this dir, delete comment out and do predict
    # predict(model, current_dir, test_dataset)


if __name__ == "__main__":
    main()
