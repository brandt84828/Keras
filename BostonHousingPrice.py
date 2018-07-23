# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:59:42 2018

@author: brandt84828
"""

from keras.datasets import boston_housing
from keras import models,layers
import numpy as np
import matplotlib.pyplot as plt
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()


#求出mean 和 std >>>normalization
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],))) #特徵數量
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))#輸出一個結果
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


k=4
num_val_samples=len(train_data) // k #兩數相除 向下取整
# =============================================================================
# num_epochs=100
# all_scores=[]
# 
# for i in range(k):
#     print('processing fold #', i)
#     # Prepare the validation data: data from partition # k
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]# i~i+1為驗證資料
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#     
#     # Prepare the training data: data from all other partitions
#     partial_train_data = np.concatenate(#合併i之前和i+1之後
#         [train_data[:i * num_val_samples],
#          train_data[(i + 1) * num_val_samples:]],
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i + 1) * num_val_samples:]],
#         axis=0)
# 
#     # Build the Keras model (already compiled)
#     model = build_model()
#     # Train the model (in silent mode, verbose=0)
#     model.fit(partial_train_data, partial_train_targets,
#               epochs=num_epochs, batch_size=1, verbose=0)#verbose=0>>不輸出標準日誌訊息
#     # Evaluate the model on the validation data
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)   
# np.mean(all_scores)
# =============================================================================

num_epochs  = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]#每一次epoch的平均值

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)#plot(x,y)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]#抓前一個
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])#取第10筆之後

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)#epochs=80開始overfitting 所以改80
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score