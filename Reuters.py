# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:32:52 2018

@author: brandt84828
"""
from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels,dimension=46):#one-hot encoding也是指說有值轉成1 其餘皆為0
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1.
    return results

one_hot_train_labels=to_one_hot(train_labels)
one_hot_test_labels=to_one_hot(test_labels)
#one_hot_test_labels = to_categorical(test_labels) #結果同上 可直接用此將類別向量>二元矩陣

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))#64是因為這次labels結果較多 如果縮太小可能會在學習過程造成資訊遺失
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))#因為output較多 這裡改用softmax 結果出來全部加總機率為1

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',#因應結果可能性較多 這裡採用categorical_crossentropy
              metrics=['accuracy'])
x_val = x_train[:1000] #用1000sample作為驗證資料
partial_x_train = x_train[1000:]
y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]


#model.fit(partial_x_train,#跑完epochs=20後 因為發現在epochs=9開始overfitting 所以model 最後改跑這段
#          partial_y_train,
#          epochs=9,#在9的時候開始overfitting
#          batch_size=512,
#          validation_data=(x_val,y_val))
#model.evaluate(x_test,one_hot_test_labels)

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)#考慮畫上去的間距才這樣而不直接輸入20
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()#清除圖
acc= history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

