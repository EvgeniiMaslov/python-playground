# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from utils import *
from sklearn.model_selection import train_test_split




path = 'IMG'
data = importDataInfo('')


data = balanceData(data, False)


imagesPath, steerings = loadData(path, data)
# print(imagesPath[0], steering[0])

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total training images: ', xTrain.shape[0])
print('Total validation images: ', xVal.shape[0])

model = createModel()
model.summary()



history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10,
          validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)



model.save('model.h5')
print('Model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Losses')
plt.xlabel('epoch')
plt.show()
