import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

from resnet import ResnetBuilder

# データセットを読み込む。
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# [0, 1] に正規化する。
train_X = train_images / 255
test_X = test_images / 255

# (H, W) -> (H, W, 1) にする。
train_X = train_X[..., np.newaxis]
test_X = test_X[..., np.newaxis]

# クラス ID とクラス名の対応
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


input_shape = (28, 28, 1)  # モデルの入力サイズ
num_classes = 10  # クラス数

# ResNet18 モデルを作成する。
model = ResnetBuilder.build_resnet_18(input_shape, num_classes)

# モデルをコンパイルする。
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 学習する。
model.fit(train_X, train_labels, epochs=5)


# テストデータに対する性能を確認する。
test_loss, test_acc = model.evaluate(test_X, test_labels)

print(f"test loss: {test_loss:.2f}, test accuracy: {test_acc:.2%}")
# test loss: 0.36, test accuracy: 86.93%

# テストデータを推論する。
predictions = model.predict(test_X)

# test_images[0] の予測結果
fig, ax = plt.subplots()
ax.imshow(test_images[0], cmap="gray")

pred = predictions[0]
for name, score in zip(class_names, pred):
    print(f"{name}: {score:.2%}")