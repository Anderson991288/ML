# CNN : 使用 ResNet 作為基礎架構
## Food Classification:
  
● The images are collected from the food-11 dataset classified into 11 classes.

● The dataset here is slightly modified:

● Training set: 280 * 11 labeled images + 6786 unlabeled images

● Validation set: 60 * 11 labeled images

● Testing set: 3347 images

## 數據預處理與數據增強
```
import torchvision.transforms as transforms
```
transforms 函式庫主要提供了一些常用的圖像轉換操作，如：圖像縮放、圖像旋轉等，用於數據預處理和數據增強。

transforms.Compose 是用來將多個圖像轉換操作組合在一起。


* transforms.Resize((128, 128)): 調整圖像大小至 128x128

* transforms.RandomHorizontalFlip(): 進行隨機水平翻轉

* transforms.RandomRotation(15): 圖像隨機旋轉，旋轉角度在-15至15度之間

* transforms.RandomCrop((120, 120)): 隨機裁剪圖像至 120x120

* transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1): 對圖像進行隨機亮度、對比度、飽和度和色相的調整

* transforms.ToTensor(): 將圖像從PIL或numpy.ndarray格式轉換為torch.Tensor格式，並將像素值由[0, 255]轉為[0.0, 1.0]

對於測試數據（test_tfm），只進行圖像大小調整和轉換為torch.Tensor格式的操作

能提高模型的泛化能力，避免模型過擬合，並且可以增加訓練數據量。


## ResNet-50
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
## ResNet-18
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

## 參數調整:
```
* epoch = 80
* batch_size = 64
```
## Code:
[CNN_ResNet_50.ipynb](https://github.com/Anderson991288/Machine-Learning/blob/main/CNN/CNN_ResNet_50.ipynb)

[CNN_ResNet_18.ipynb](https://github.com/Anderson991288/Machine-Learning/blob/main/CNN/CNN_ResNet_18.ipynb)

## Result:
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
