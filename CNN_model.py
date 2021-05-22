from tensorflow.keras import datasets, layers, optimizers, Sequential

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data("mnist.pkl")
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    # 输入层，将数据整形为特定格式 28 × 28 × 1 进行处理
    layers.InputLayer(input_shape=(28, 28)),
    layers.Reshape((28, 28, 1)),

    # 第一卷积层
    # 卷积核大小为 3，步距为 1，选择 SAME 填充，过滤器个数为 16
    layers.Conv2D(kernel_size=3, strides=1, filters=16, padding='same'),
    # BN减少梯度消失，加快了收敛过程。起到类似dropout一样的正则化能力，一定程度上防止过拟合。
    layers.BatchNormalization(),
    # 激活函数为 ReLU
    layers.ReLU(),
    # 池化层，选择 2 × 2 最大池化
    layers.MaxPooling2D(pool_size=2, strides=2),

    # 第二卷积层，与第一层相似
    # 过滤器个数为 32
    layers.Conv2D(kernel_size=3, strides=1, filters=32, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=2, strides=2),

    # 将卷积层输出结果（四维）通过 Flatten 降到一维
    layers.Flatten(),

    # 第一全连接层，使用 ReLU 作为激活函数
    layers.Dense(units=128),
    layers.BatchNormalization(),
    layers.ReLU(),

    # 第二全连接层，使用 softmax 对输出结果进行归一化
    layers.Dense(units=10, activation='softmax')
])

# 选择自适应梯度下降（Adam）和交叉熵损失函数
model.compile(
    optimizer=optimizers.Adam(lr=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练 100 轮
model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    validation_data=(x_test, y_test)
)
