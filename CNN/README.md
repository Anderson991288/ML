# CNN : 使用 ResNet-50 作為基礎架構
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

# 參數調整:
```
* epoch = 80
* batch_size = 64
```
* [Code](https://github.com/Anderson991288/Machine-Learning/blob/main/CNN/CNN_ResNet_50.ipynb)

# Result :
```
* 訓練: 損失 = 0.26892, 準確率 = 0.91167
* 驗證: 損失 = 1.73879, 準確率 = 0.60824
```

