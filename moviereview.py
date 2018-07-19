# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:09:26 2018

@author: brandt84828
"""

from keras.datasets import imdb
from keras import models
from keras import layers
import  numpy as np
import matplotlib.pyplot as plt
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000) #只保留頻率TOP 10000的字
#[max(sequence) for sequence in train_data]#找每筆data內文字index最大
#max([max(sequence) for sequence in train_data])#整個文件內文字的index最大值
word_index = imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])#反轉dictionary
decode_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])#前3個index分別為 padding  start of sequence unknown >>>and get just to get dict(如果找不到key就預設為?)

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))#建一個sequences長度*dimension的全部0 tensor
    for i,sequence in enumerate(sequences):#在每一列中=有數值該欄位轉換成1
        results[i,sequence] = 1.
    return results

x_train=vectorize_sequences(train_data) #向量化
x_test=vectorize_sequences(test_data)
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

model=models.Sequential()#使用順序模型
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
# 現在模型就以尺寸為 (*,10000) 的數組作為输入
# 其输出數組的尺寸為 (*, 16)
model.add(layers.Dense(16,activation='relu'))
#第一層之後就不用輸入尺寸了 會以上一層的輸出進行
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', #這3個皆可自訂 在上面import optimizers losses metrics
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val=x_train[:10000]#創建驗證集 0到9999項目(即是取前10000項)
partial_x_train=x_train[10000:]#從10000+1後開始取到結束
y_val=y_train[:10000]
partial_y_train=y_train[10000:]

history=model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
history_dict = history.history#可以看資料和驗證的準確度 損失(出來是一個字典型態)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)#range(1,21)
plt.plot(epochs,loss_values,'bo',label='Training loss')#參數分別為X,Y data ,bo is blue dot,label=圖例
plt.plot(epochs,val_loss_values,'b',label='Validation loss')#b=solid blue line
plt.title('Training and validation loss')#圖的標題
plt.xlabel('Epochs')#x座標
plt.ylabel('Loss')#y座標
plt.legend()#設置圖例(值用上面plot輸入)
plt.show()

plt.clf()#清除圖 防止與前一個圖重疊
acc = history_dict['acc']
val_acc=history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size=512)#epoch20>4 執行20之後會發現驗證資料在4的表現最好>超過之後有overfitting的疑慮(為了緩和overfitting在此改成取4)
results = model.evaluate(x_test,y_test)
model.predict(x_test)#預測y