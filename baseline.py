import numpy as np 
import torch
import torchvision as ptv
from torch.utils.data import Dataset

class MLP(torch.nn.Module):
	def __init__(self):
		super(MLP,self).__init__()
		self.fc1 = torch.nn.Linear(512,256)
		self.fc2 = torch.nn.Linear(256,128)
		self.fc3 = torch.nn.Linear(128,10)
	def forward(self,din):
		dout = torch.nn.functional.relu(self.fc1(din))
		dout = torch.nn.functional.relu(self.fc2(dout))
		return torch.nn.functional.softmax(self.fc3(dout))

class MyDataset(Dataset):
	def __init__(self, d, l):
		self.n_samples = td.shape[0]
		self.x_data = td
		self.y_data = tl.flatten()
	def __getitem__(self, index):
		sample = self.x_data[index], self.y_data[index]
		return sample
	def __len__(self):
		return self.n_samples

dataset = np.load("course_train.npy")
D = np.delete(dataset, 512, axis=1)
permutation = np.random.permutation(D.shape[0])
shuffled = D[permutation, :]
training = shuffled[:3000,:].copy()
testing = shuffled[3000:,:].copy()

td = torch.from_numpy(training[:,:512]).float()
tl = torch.from_numpy(training[:,512:]).long()
sd = torch.from_numpy(testing[:,:512]).float()
sl = torch.from_numpy(testing[:,512:]).long()



dst = MyDataset(d = td, l = tl)
dss = MyDataset(d = sd, l = sl)
trainLoader = torch.utils.data.DataLoader(dst, batch_size=1)
testLoader = torch.utils.data.DataLoader(dss, batch_size=1)


model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_func = torch.nn.CrossEntropyLoss()  

for x in range(15):
	print("epoch: ", x)
	cnt = 0
	cor = 0
	for i,data in enumerate(trainLoader):
		optimizer.zero_grad()
		(inputs,labels) = data
		inputs = torch.autograd.Variable(inputs)
		labels = torch.autograd.Variable(labels)
		outputs = model(inputs)
		cnt+=1
		if(torch.argmax(outputs)==labels):
			cor+=1
		loss = loss_func(outputs,labels)
		loss.backward()
		optimizer.step()
	print("accu: ", cor/cnt)

c = 0
r = 0
for i,data in enumerate(trainLoader):
	(inputs,labels) = data
	inputs = torch.autograd.Variable(inputs)
	labels = torch.autograd.Variable(labels)
	outputs = model(inputs)
	c+=1
	if(torch.argmax(outputs)==labels):
			r+=1

print("Testing Accuracy:",r/c)