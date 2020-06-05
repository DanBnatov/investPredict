import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import task1
import task3

from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler



#I haven't used the GPU functionality because I dont have CUDA GPU
#The gpu support will be added later


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1000, kernel_size=300, stride=5, dilation=2)

        self.relu1 = nn.ReLU()

        #self.conv2 = nn.Conv1d(in_channels=3000, out_channels=1000, kernel_size=50, stride=1, dilation=1)

        #self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool1d(3, 3)

        self.fc1 = nn.Linear(27, 3)


    def forward(self, input):

        convOut = self.conv1(input.view(1, 1, 1000))#input.view(1000, 1, -1))

        out = self.relu1(convOut)

        #out = self.conv2(out)#input.view(1000, 1, -1))

        #out = self.relu2(out)

        out = self.maxpool(out)

        out = self.fc1(out)

        return out


X, Ymin, Ymax = task1.generateDataset(1000, 17, 1024)

model = ConvNet()
learning_rate = 0.000001
num_epochs = 5000
loss_fn = torch.nn.CrossEntropyLoss()

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimiser = optim.SGD(model.parameters(), lr=learning_rate)#, momentum=0.9)


k = 16

for t in range(num_epochs):

    Xmini, Ymm = task3.setUpTensor(X, Ymin, Ymax, k, 1000)

    for i in range(k):

        y_pred = model(Xmini[i])

        loss = loss_fn(y_pred[0], Ymm[i])

        if i == 15:

            print("Epoch ", t, "Loss: ", loss.item())

            #Clear gradient
            optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()




##################
####TEST MODEL####
##################
##################

Xt, Ymint, Ymaxt = generateDataset(1000, 17, 1024)
Xminit, Yminminit = setUpTensor(Xt, Ymint, Ymaxt, 16, 1000)

from sklearn.metrics import confusion_matrix

prediction = model(Xminit[1])

soft = nn.Softmax(dim = 1)

prob = soft(prediction)

print(prob)

preds = np.argmax(prob.detach().numpy(), axis=2)

print(preds[0])

print(Yminminit[1].detach().numpy())

confusion_matrix(preds[0], Yminminit[1].detach().numpy())
