import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


df = pd.read_csv('outputs2.csv')
x = np.array(df.iloc[:, 0]).flatten()
accu = np.array(df.iloc[:, 1]).flatten()
loss = np.array(df.iloc[:, 2]).flatten()

plt.plot(x, accu, label='Simple Mean')

df = pd.read_csv('outputs1.csv')
x = np.array(df.iloc[:, 0]).flatten()
accu = np.array(df.iloc[:, 1]).flatten()
loss = np.array(df.iloc[:, 2]).flatten()

plt.plot(x, accu, label='CGC')


plt.ylim(0, 1)
plt.xscale('log')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy under Bitflip Attack")
plt.show()




df = pd.read_csv('outputs2.csv')
loss = np.array(df.iloc[:, 2]).flatten()
plt.plot(x, loss, label='Simple Mean')

df = pd.read_csv('outputs1.csv')
loss = np.array(df.iloc[:, 2]).flatten()
plt.plot(x, loss, label='CGC')

plt.xscale('log')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss under Bitflip Attack")
plt.show()