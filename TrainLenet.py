# from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score: Import các hàm đánh giá hiệu suất từ thư viện scikit-learn để tính ma trận nhầm lẫn, độ chính xác, recall, precision và F1-score.
# from numpy import argmax: tìm chỉ số của giá trị lớn nhất trong một mảng.
# import numpy as np:  làm việc với các mảng và ma trận.
# import pandas as pd: làm việc với dữ liệu dưới dạng DataFrame.
# import matplotlib.pyplot as plt:trực quan hóa dữ liệu.
# import os:  đọc/ghi file.

# from sklearn.model_selection import train_test_split: chia tập dữ liệu thành tập huấn luyện và tập kiểm tra.
# from keras.models import Sequential: Import class Sequential từ thư viện Keras để xây dựng mô hình mạng nơ-ron tuần tự.
# from keras.layers import Dense, Dropout, Activation, Flatten: Import các lớp cần thiết từ thư viện Keras để xây dựng mô hình mạng nơ-ron.
# from keras.layers import Convolution2D, MaxPooling2D: Import các lớp cần thiết cho việc xây dựng mạng CNN từ thư viện Keras.
# from tensorflow.keras import utils: Import module utils từ thư viện Keras để chuyển đổi nhãn sang dạng one-hot encoding.

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from numpy import argmax
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import utils
import tensorflow as tf
from tensorflow import keras

# PATH
path = "./train_lenet/"
categories = ['man', 'woman']

# ============================= RESIZE =================================

data = []  # dữ liệu
labels = []  # nhãn
imagePaths = []
HEIGHT = 64
WIDTH = 64
# 24 24
N_CHANNELS = 3

# =========================== Xáo trộn ảnh ===================================

# Duyệt qua mỗi nhãn trong danh sách categories.
# Duyệt qua từng file ảnh trong thư mục tương ứng với nhãn đó.
# Mỗi ảnh sẽ được thêm vào imagePaths với đường dẫn đầy đủ và nhãn tương ứng (được biểu diễn dưới dạng số).
# Sau khi duyệt xong, danh sách imagePaths sẽ được xáo trộn ngẫu nhiên để tránh việc mô hình học theo thứ tự.
for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k])

random.shuffle(imagePaths)
# print(imagePaths[:10])

# ======================= Tiền xử lý=======================================

# Đọc và tiền xử lý ảnh:
# Mỗi ảnh trong imagePaths được đọc bằng OpenCV và thay đổi kích thước thành kích thước cố định (HEIGHT x WIDTH).
# Ảnh được chuẩn hóa bằng cách chia tất cả các giá trị pixel cho 255 để đưa chúng về phạm vi từ 0 đến 1.
# Ảnh được thêm vào danh sách data.
# Nhãn của ảnh được lấy từ phần tử thứ hai của imagePath và thêm vào danh sách labels.

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    label = imagePath[1]
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Trực quan hóa dữ liệu:
# Một grid 3x4 subplot được tạo ra để hiển thị một số ảnh từ dữ liệu đã được tiền xử lý.
# Duyệt qua 12 ảnh đầu tiên trong data và hiển thị chúng trong các subplot.
# Mỗi subplot được đặt tiêu đề là nhãn tương ứng với ảnh đó (được chuyển đổi từ dạng số sang nhãn thông qua danh sách categories và labels).
plt.subplots(3, 4)
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
# plt.show()

# ============================Chia dữ liệu==================================
# Các tham số đầu vào là data (dữ liệu) và labels (nhãn), test_size=0.2 chỉ định tỷ lệ dữ liệu được sử dụng cho tập kiểm tra (20% trong trường hợp này), và random_state=42 là một giá trị nguyên được sử dụng để khởi tạo trạng thái ngẫu nhiên cho quá trình chia dữ liệu.
# Kết quả trả về là trainX (dữ liệu huấn luyện), testX (dữ liệu kiểm tra), trainY (nhãn của dữ liệu huấn luyện), và testY (nhãn của dữ liệu kiểm tra).
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42)  # random_state=30)

# chuyển đổi các nhãn từ dạng số sang dạng one-hot encoding. mỗi nhãn được biểu diễn bằng một vector nhị phân với độ dài bằng số lượng lớp, trong đó chỉ có một phần tử là 1 và các phần tử còn lại là 0
trainY = utils.to_categorical(trainY, len(categories))

# ===========================huan luyen===================================
# Ban đầu EPOCHS = 20, BS = 32.
# EPOCHS = 20: Số lần mà toàn bộ tập huấn luyện sẽ được truy cập và huấn luyện mạng nơ-ron.  Batch size nhỏ có thể giúp quá trình huấn luyện ổn định hơn và tránh rơi vào điểm cực tiểu cục bộ.
# INIT_LR = 1e-3: Tốc độ học ban đầu của mô hình.
# BS = 16: Kích thước batch, tức số lượng mẫu dữ liệu được sử dụng trong mỗi lần cập nhật trọng số của mạng nơ-ron.
EPOCHS = 20
INIT_LR = 1e-3
BS = 16

# Biến class_names được gán bằng categories, danh sách các lớp (trong trường hợp này là ['man', 'woman']), để sử dụng sau này khi hiển thị các kết quả hoặc đánh giá mô hình.
class_names = categories

# Xây dựng mạng nơ-ron:
# Sử dụng lớp Sequential để tạo một mô hình mạng nơ-ron tuần tự.
# Thêm lớp Convolutional đầu tiên với 32 bộ lọc (filters) có kích thước 5x5, hàm kích hoạt relu, và sử dụng padding 'same' để giữ nguyên kích thước đầu vào. input_shape là kích thước của một ảnh đầu vào, có chiều cao (HEIGHT), chiều rộng (WIDTH), và 3 kênh màu (RGB).
# Tiếp theo là lớp MaxPooling để giảm kích thước của feature maps.
# Thêm lớp Convolutional thứ hai với 48 bộ lọc có kích thước 5x5, hàm kích hoạt relu, và sử dụng padding 'valid'.
# Tiếp tục là lớp MaxPooling để giảm kích thước của feature maps.
# Lớp Flatten được sử dụng để chuyển từ tensor 2D sang tensor 1D trước khi đưa vào các lớp fully connected.
# Thêm hai lớp Dense với kích thước lần lượt là 256 và 84, hàm kích hoạt relu.
#       Dense(256, activation='relu'): Đây là lớp fully connected với 256 neuron. Mỗi neuron trong lớp này sẽ kết nối đến tất cả các neuron trong lớp trước đó (ở trạng thái flatten).
#       activation='relu': Đây là hàm kích hoạt được sử dụng cho các neuron trong lớp này, trong trường hợp này là hàm kích hoạt ReLU (Rectified Linear Unit). Hàm ReLU được chọn phổ biến trong các mô hình mạng nơ-ron vì khả năng giảm thiểu hiện tượng biến mất đạo hàm và tăng tốc độ học.
# Lớp Dense cuối cùng có số lượng neuron bằng số lượng lớp trong class_names, và sử dụng hàm kích hoạt softmax để tính toán xác suất dự đoán cho mỗi lớp.
model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same',
          activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D(strides=2))
model.add(Convolution2D(filters=48, kernel_size=(
    5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
# Trích thông tin
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
# Tính toán xác suất dự đoán cho mỗi lớp.
model.add(Dense(len(class_names), activation='softmax'))

# Chọn hàm loss là categorical_crossentropy, một hàm mất mát thích hợp cho bài toán phân loại nhiều lớp.
# Sử dụng thuật toán tối ưu hóa adam với tốc độ học là INIT_LR.
# Đánh giá hiệu suất của mô hình dựa trên độ chính xác (accuracy).
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Sử dụng phương thức fit() để huấn luyện mô hình trên tập dữ liệu huấn luyện (trainX, trainY).
model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)

model.save("gender_detect_lenet.h5")
# ==========================kiem tra su dung cua mo hinh====================================

# Dự đoán nhãn:
# pred = model.predict(testX): Sử dụng phương thức predict() của mô hình để dự đoán nhãn của dữ liệu kiểm tra (testX). Kết quả trả về là một mảng 2D chứa xác suất dự đoán cho mỗi mẫu dữ liệu và cho mỗi lớp.

pred = model.predict(testX)
# Chuyển đổi dự đoán thành nhãn:
# predictions = argmax(pred, axis=1): Dùng hàm argmax() từ thư viện numpy để tìm chỉ số của giá trị lớn nhất trong mỗi hàng của mảng dự đoán pred. Điều này sẽ cho chúng ta biết lớp dự đoán cho mỗi mẫu dữ liệu.
predictions = argmax(pred, axis=1)  # return to label

# Tạo ma trận nhầm lẫn:
# confusion_matrix(testY, predictions): Tính ma trận nhầm lẫn bằng cách so sánh nhãn thực tế (testY) và nhãn dự đoán (predictions). Ma trận nhầm lẫn là một cách để đánh giá hiệu suất của mô hình, nó hiển thị số lượng các dự đoán đúng và sai trong từng lớp.
cm = confusion_matrix(testY, predictions)

# Trực quan hóa ma trận nhầm lẫn:
# Tạo một hình ảnh mới (fig) và thêm một trục subplot (ax) vào đó.
# Sử dụng hàm matshow() để hiển thị ma trận nhầm lẫn dưới dạng một hình ảnh màu.
# Tiêu đề của hình ảnh được đặt là 'Model confusion matrix'.
# Thêm thanh màu (colorbar) để biểu thị giá trị của các ô trong ma trận.
# Đặt nhãn của các trục X và Y là các danh mục (categories).
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Model confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + categories)
ax.set_yticklabels([''] + categories)

# sử dụng một vòng lặp để thêm văn bản vào các ô của ma trận nhầm lẫn trên đồ thị.
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(i, j, cm[j, i], va='center', ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Độ chính xác được định nghĩa là tỷ lệ giữa số lượng dự đoán đúng và tổng số lượng mẫu dữ liệu.
accuracy = accuracy_score(testY, predictions)
print("Accuracy : %.2f%%" % (accuracy*100.0))
print("\n")
# ----------------------------------------------
# Recall là tỷ lệ giữa số lượng dự đoán đúng cho mỗi lớp và tổng số lượng mẫu dữ liệu thuộc lớp đó.
recall = recall_score(testY, predictions, average='weighted')
print("Recall :%.2f%%" % (recall*100))
print("\n")
# ----------------------------------------------
# Precision là tỷ lệ giữa số lượng dự đoán đúng cho mỗi lớp và tổng số lượng mẫu được dự đoán là thuộc lớp đó.
precision = precision_score(testY, predictions, average='weighted')
print("Precision : %.2f%%" % (precision*100.0))
print("\n")
# ----------------------------------------------

# F1-score là một phép đo kết hợp giữa precision và recall, và thường được sử dụng để đánh giá hiệu suất của một mô hình phân loại. F1-score là một số từ 0 đến 1, trong đó giá trị càng cao thể hiện một mô hình phân loại tốt hơn.
f1 = f1_score(testY, predictions, average='weighted')
print("F1 : %.2f%%" % (f1*100.0))
print("\n")
