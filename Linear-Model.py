from tensorflow.keras import datasets, layers, optimizers, Sequential

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data("mnist.pkl")
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
  # 定义输入格式，通过Flatten层降至一维
  layers.Flatten(input_shape=(28, 28)),
  # 第一全连接层，选择ReLU作为激活函数
  layers.Dense(128, activation='relu'),
  # 使用防止过拟合
  layers.Dropout(0.2),
  # 第二全连接层，使用softmax对输出进行归一化
  layers.Dense(10, activation='softmax')
])

# 选择自适应梯度下降（Adam）和交叉熵误差函数
model.compile(optimizer=optimizers.Adam(lr=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练5轮
model.fit(x_train, y_train, epochs=5)
# 测试模型准确率
model.evaluate(x_test,  y_test, verbose=2)
