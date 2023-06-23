# CNN : 使用 ResNet 作為基礎架構
### Food Classification:
  
● The images are collected from the food-11 dataset classified into 11 classes.

● The dataset here is slightly modified:

● Training set: 280 * 11 labeled images + 6786 unlabeled images

● Validation set: 60 * 11 labeled images

● Testing set: 3347 images

### ResNet-50
```
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 使用 ResNet-50 模型作為基礎架構
        self.model = models.resnet50(pretrained=False)

        # 替換最後一層全連接層
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 11)

    def forward(self, x):
        x = self.model(x)
        return x
```
# ResNet-18
```
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # ResNet-18 模型
        self.resnet = models.resnet18(pretrained=False)

        # 替換最後一層全連接層
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 11)

    def forward(self, x):
        x = self.resnet(x)
        return x
```

# 參數調整:
```
* epoch = 80
* batch_size = 64
```
# Code:
[CNN_ResNet_50.ipynb](https://github.com/Anderson991288/Machine-Learning/blob/main/CNN/CNN_ResNet_50.ipynb)
[CNN_ResNet_18.ipynb](https://github.com/Anderson991288/Machine-Learning/blob/main/CNN/CNN_ResNet_18.ipynb)

# Result:
* ResNet-50
  
```
* 訓練: 損失 = 0.26892, 準確率 = 0.91167
* 驗證: 損失 = 1.73879, 準確率 = 0.60824
```

* ResNet-18
  
```
* 訓練 損失 = 0.15401, 準確率 = 0.94834
* 驗證 損失 = 2.48382, 準確率 = 0.49006
```
