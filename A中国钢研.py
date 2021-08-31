#导入需要使用的库
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import os
#设置字体为楷体
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False#显示负号

data_path = '数据集C (含列名).csv'
datas = pd.read_csv(data_path)
print(datas.head())
print('\n 全部列名 \n',datas.columns)
#我们取出最后一列的前1000条记录来进行预测
Y = datas['Y'][:1000]
x = np.arange(len(Y))    #获得变量x，它是1，2，……，50
y = np.array(Y)    # 将counts转成预测变量（标签）：y
# 将所有的数据集分为测试集和训练集，后20%为测试集
train_df = datas.sample(frac=0.8,random_state=0,axis=0,replace=False)#n是选取的条数，frac是选取的比例，replace是可不可以重复选，
# weights是权重，random_state是随机种子，axis为0是选取行，为1是选取列。
test_df = datas[~datas.index.isin(train_df.index)]
print(len(test_df))
print(len(train_df))
#分特征与目标列
target_fields=['Y']
features, targets = train_df.drop(target_fields, axis=1), train_df[target_fields]
test_features, test_targets = test_df.drop(target_fields, axis=1), test_df[target_fields]
# 将数据从pandas dataframe转换为numpy
X = features.values
Y = targets['Y'].values
Y = Y.astype(float)
Y = np.reshape(Y, [len(Y),1])
losses = []
testlosses=[]

class mape_loss_func(torch.nn.Module):
    def __init__(self):
        super(mape_loss_func, self).__init__()
    def forward(self,y_pred,y):
        return torch.mean(abs(y-y_pred)/(y))*100


class neu(torch.nn.Module):
    def __init__(self):
        super(neu, self).__init__()
        self.linear1 = torch.nn.Linear(5,64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 256)
        self.linear4 = torch.nn.Linear(256, 128)
        self.linear5 = torch.nn.Linear(128,64)
        self.linear6 = torch.nn.Linear(64, 1)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        x = self.activate(self.linear5(x))
        x = self.activate(self.linear6(x))

        return x
batch_size =128
neu = neu()
cost = mape_loss_func()# mean代表均值，sum代表求和
optimizer = torch.optim.Adam(neu.parameters(), lr=0.007,weight_decay=1e-5)
# 神经网络训练循环
for i in range(10000):
    # 每128个样本点被划分为一个撮，在循环的时候一批一批地读取
    batch_loss = []
    # start和end分别是提取一个batch数据的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = Variable(torch.FloatTensor(X[start:end]))
        yy = Variable(torch.FloatTensor(Y[start:end]))
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # 每隔100步输出一下损失值（loss）
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 打印输出损失值
plt.plot(np.arange(len(losses)) * 100, losses)
plt.xlabel('训练次数')
plt.ylabel('相对误差')
plt.show()


targets = train_df['Y'] #读取训练集的Y数值
targets = targets.values.reshape([len(targets),1]) #将数据转换成合适的tensor形式
targets = targets.astype(float) #保证数据为实数

# 将属性和预测变量包裹在Variable型变量中
x = Variable(torch.FloatTensor(features.values))

y = Variable(torch.FloatTensor(targets))

# 用神经网络进行预测
predict = neu(x)
predict = predict.data.numpy()
fig, ax = plt.subplots(figsize = (10, 7))

ax.plot(predict , label='训练预测值')
ax.plot(targets, label='原始值')
ax.legend()
ax.set_xlabel('样本编号')
ax.set_ylabel('Y')
plt.show()




#测试神经网络
# 用训练好的神经网络在测试集上进行预测

targets = test_targets['Y'] #读取测试集的Y数值
targets = targets.values.reshape([len(targets),1]) #将数据转换成合适的tensor形式
targets = targets.astype(float) #保证数据为实数

# 将属性和预测变量包裹在Variable型变量中
x = Variable(torch.FloatTensor(test_features.values))
y = Variable(torch.FloatTensor(targets))
# 用神经网络进行预测
predict = neu(x)
predict = predict.data.numpy()
fig, ax = plt.subplots(figsize = (10, 7))

ax.plot(predict , label='测试集预测值')
ax.plot(targets, label='原始值')
ax.legend()
ax.set_xlabel('样本编号')
plt.show()

for i in range(100):
    # 每128个样本点被划分为一个撮，在循环的时候一批一批地读取
    test_loss = []
    # start和end分别是提取一个batch数据的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = Variable(torch.FloatTensor(X[start:end]))
        yy = Variable(torch.FloatTensor(Y[start:end]))
        predict = neu(xx)
        loss = cost(predict, yy)
        test_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(test_loss))
        print(i, np.mean(test_loss))
plt.plot(np.arange(len(testlosses)) * 100, testlosses)
plt.xlabel('次数')
plt.ylabel('相对误差')
plt.show()
# 打印输出损失值

#利用神经网络进行预测
evaldata = pd.read_csv('测试集C(含列名).csv')
print(datas.head())
print('\n 全部列名 \n',datas.columns)
#我们取出最后一列的前1000条记录来进行预测
X1 = datas['X1'][:200]
print(len(evaldata))
x = np.arange(len(X1))    #获得变量x，它是1，2，……，50
target_fields=['Y']
features = evaldata.drop(target_fields, axis=1)
# 将数据从pandas dataframe转换为numpy
x = Variable(torch.FloatTensor(features.values))
predict = neu(x)
predict = predict.data.numpy()
print(predict)

