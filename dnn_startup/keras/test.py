# 参考: https://qiita.com/wataoka/items/5c6766d3e1c674d61425

##################################
# データのロード
##################################
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils.np_utils import to_categorical


# 画像を1次元化
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 画素を0~1の範囲に変換(正規化)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 正解ラベルをone-hot-encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


##################################
# モデル定義
##################################
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=784))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train,
            batch_size=100,
            epochs=12,
            verbose=1)