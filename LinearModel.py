#1. import torch and numpy and all other libraries
import torch
from torch import nn
import numpy as np
import matplotlib as plt
from pathlib import Path

#2. We need to prepare data as tensors
KnownWeight = 0.9
KnownBias = 0.2
# arbitrary values, different from last ones
start = 0
end = 1
step = 0.05
X = torch.arange(start,end,step).unsqueeze(1) # create a sample input matrix, which holds values from 0 to 1
Y = KnownWeight * X + KnownBias # our desired Y

trainSplit = int(0.8 * len(X))

# Note, the code below will run only on cuda, change lines referring it to make it work on other devices
XTrain = X[:trainSplit] # all data until 80% mark
XTest = X[trainSplit:]

YTrain = Y[:trainSplit]
YTest = Y[trainSplit:]

import matplotlib.pyplot as  plt
def plot_projections(train_data=XTrain,train_labels=YTrain,test_data=XTest,test_labels=YTest,predictions=None):
  # plots training data, test data and compares predictiosn
  plt.figure(figsize=(10,7))

  plt.scatter(train_data.cpu(),train_labels.cpu(),c="b",s=4,label="Training data")

  plt.scatter(test_data.cpu(),test_labels.cpu(),c="g",s=4,label="Testing Data")


  if(predictions != None):
    plt.scatter(test_data.cpu(),predictions.cpu(),c="r",s=4,label="Predictions")
  plt.legend(prop={"size":14});
# training and testing sets ready to go

# now proceed with setting up a model
#3. Create the model class
class LinearRegressionModelTwo(nn.Module):
  def __init__(self):
    super().__init__()
    # use nn linear for parameters
    self.linear_layer = nn.Linear(in_features=1, # 1 piece of data comes in
                                  out_features=1) # 1 piece of data comes out

    # this just implements a linear transform, see torch.linear transform

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.linear_layer(x)

torch.manual_seed(100)

model1 = LinearRegressionModelTwo()

#4. Setup loss and optimator function

lossFunction = nn.L1Loss()

optimFunction = torch.optim.SGD(params=model1.parameters(),lr=0.0005) # this works
# now we can construct the traiining loop
# and we can connect to the gpu to make it run faster
device = "cuda"
next(model1.parameters()).device # check the model
model1.to(device)
next(model1.parameters()).device # check the model Now its cuda! running on a gpu

#5. Training code: Loop

epochs = 20000
epochCount = []
TrainingLoss = []
TestLoss = []

XTrain = XTrain.to(device)
XTest = XTest.to(device)

YTrain = YTrain.to(device) # puts the tensors on the gpu
YTest = YTest.to(device)
for epoch in range(epochs):
  # set the model to training mode and make it eval
  model1.train() #
  yPredsTrain = model1(XTrain) # forward pass
  currentLoss = lossFunction(yPredsTrain,YTrain) # this calculates the loss between actual results and desired results
  if(epoch % 10 == 0):
    epochCount.append(epoch)
    TrainingLoss.append(currentLoss)


  optimFunction.zero_grad() # make grads none, to prevent gradient stacking
  currentLoss.backward()
  optimFunction.step()

  # test the model
  model1.eval() # enter eval mode, for performance reasons
  with torch.inference_mode(): #turn off gradient tracking
    yPredsTest = model1(XTest)
    currentLoss = lossFunction(yPredsTest, YTest)

    if(epoch % 10000 == 0):
      TestLoss.append(f"{currentLoss}")
      epochCount.append(f"{epoch}")
      print(f" Epoch Count: {epochCount[-1]},Training Loss: {TrainingLoss[-1]}, Test Loss: {TestLoss[-1]}")


with torch.inference_mode():
  plot_projections(predictions=yPredsTest)

    # now plot it


print(f"Parameters: {model1.state_dict()}")


# now lets save it

MODELPATH = Path("models")

MODELPATH.mkdir(parents=True,exist_ok=True)

MODELNAME= "PYTORCH MODEL 1: LINEAR REGRESSION.pth"
MODELSAVEPATH = MODELPATH/MODELNAME

# save the model state dict
torch.save(model1.state_dict,MODELSAVEPATH)


