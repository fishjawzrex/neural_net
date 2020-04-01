import mnist,numpy as np
from matplotlib import pyplot as plt

a = np.array([[1,2],[3,4]])
b = np.array([[1],[2],[3],[4],[1],[2],[7],[2]])

class Net:
  def __init__(self):
    self.weights = []
    self.biases = []
    self.L = self.createLayers()
    self.z_s = None
    self.a_s = None
    self.success=0
    self.trials=10
    self.popWeightsBiases()
    self.x=[]
    self.y=[]

  def createMbs(self,arr, size):
    n = len(arr)
    m = size
    mbs = []
    k = 0
    while k<=n-m:
      mbs += [arr[k:m+k]]
      k+=m
    return mbs

  def createLayers(self,midLayers = [100,50,25,15]):
    L = []
    x= mnist.train_images()[0].flatten().reshape(784,1)
    L.append(np.zeros(x.shape))
    for i in midLayers:
      L.append(np.zeros((i,1)))
    L.append(np.array([[0.0]for i in range(10)]))
    return L

  def popWeightsBiases(self):
    L=self.L
    n=len(L)
    for i in range(n-1):
      self.weights.append(np.random.rand(len(L[i]),len(L[i+1])).transpose())
      self.biases.append(np.random.rand(len(L[i+1]),1))
    self.z_s = L[1:]
    self.a_s = L
    return
    
  def sigmoid(self,z):
    return [1/(1+np.exp(-elem)) for elem in z]

  def sp(self,z):
    z = np.array(z)
    return self.sigmoid(z)*(np.ones(z.shape)-self.sigmoid(z))

  def ff(self,x):
    self.a_s[0] = x.flatten().reshape(784,1)
    for i in range(len(self.L)-1):
      self.z_s[i] = np.dot(self.weights[i],self.a_s[i]) + self.biases[i]
      self.a_s[i+1]=self.sigmoid(self.z_s[i])
    return self.a_s[-1]

  def backprop(self,x,y):
    nw = [np.zeros(w.shape) for w in self.weights]
    nb = [np.zeros(b.shape) for b in self.biases]
    res = self.ff(x)
    res_index = np.where(res==max(res))[0][0]
    output = np.zeros((10,1))
    output[res_index-1]=1
    print(res_index,y)
    if res_index==y:
      self.success+=1
      self.y.append(self.success)
    self.trials+=1
    self.x.append(self.trials)
    ground_truth = np.zeros((10,1))
    ground_truth [y-1]=1
    cost = abs(np.subtract(res,ground_truth))
    delta = cost
    n=len(self.L)
    for i in range(n-2,-1,-1):
      nw[i] = (delta*self.sp(self.z_s[i])).dot(np.array(self.a_s[i]).transpose())
      nb[i] = delta*self.sp(self.z_s[i])
      delta = np.dot(delta.transpose(),self.weights[i]).transpose()  
    return nw,nb

  def updatewsnbs(self,a,b, eta=.15):
    nw = [np.zeros(w.shape) for w in self.weights]
    nb = [np.zeros(b.shape) for b in self.biases]
    m = len(a)
    for x,y in zip(a,b):
      dw,db = self.backprop(x,y)
      nw = [w+dnw for w,dnw in zip(self.weights,dw)]
      nb = [b+dnb for b,dnb in zip(self.biases,db)]
      self.weights = [(w-eta/m)*dnw for w,dnw in zip(self.weights,nw)]
      self.biases = [(b-eta/m)*dnb for b,dnb in zip(self.biases,nb)]
  def train(self):
    mbs = 10
    x = mnist.train_images()
    input_batches = self.createMbs(x,mbs)
    y = mnist.train_labels()
    output_batches = self.createMbs(y,mbs)
    #train network
    epoch=1
    for x,y in zip(input_batches,output_batches):
      print("epoch: {0:4d}, performance ratio: {1:}/{2:}".format(epoch,self.success,self.trials))
      self.updatewsnbs(x,y)
      epoch+=1
    self.plot()

  def plot(self):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(self.x,self.y)
    ax.show()
    return

  def test(self,x,y):
    pass

import unittest

class TestNetwork(unittest.TestCase):
  def test_len_layer(self):
    self.assertEqual(len(N.L[0]),784)
    self.assertEqual(len(N.L[1]),100)
    self.assertEqual(len(N.L[2]),50)
    self.assertEqual(len(N.L[3]),25)
    self.assertEqual(len(N.L[4]),15)
    self.assertEqual(len(N.L[5]),10)
    self.assertEqual(len(N.L),6)
    return
    
  def test_weights(self):
    self.assertEqual(len(N.weights),5)
    return

  def test_as(self):
    self.assertEqual(len(N.a_s),6)
    return

  def test_zs(self):
    self.assertEqual(len(N.z_s),5)
    return

N = Net()
# unittest.main()
N.train()
  
