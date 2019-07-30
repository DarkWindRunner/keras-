# keras-
#keras实现简单线性回归
#--------------------------------------
#准备数据
import numpy as np
np.random.seed() 
X1 = np.linspace(0, 10, 320) #在返回（0, 10）范围内的等差序列
X2 = np.linspace(0, 10, 320) #在返回（0, 10）范围内的等差序列，实际上没有用到。留着自己测试用
np.random.shuffle(X1)
Y = 0.5 * X1 + 2 + np.random.normal(0, 0.05, (320, ))
import matplotlib.pyplot as plt
plt.scatter(X1, Y)
from keras import models
model = Sequential()
from keras.layers import Dense
model.add(Dense(input_dim=1, units=1))
model.compile(loss='mse', optimizer='sgd')
X_train, Y_train = X1[:160], Y[:160] 
X_test, Y_test = X1[160:], Y[160:]  
model.fit(X_train,Y_train,batch_size=100,epochs=1000)
w,b = model.layers[0].get_weights()
print('w:',w,'b:',b)
Y_pred = model.predict(X_test)
Y_pred=Y_pred.reshape(160)
plt.scatter(X_test,Y_test)
plt.plot(X_test,Y_pred,'r-',lw=3)




