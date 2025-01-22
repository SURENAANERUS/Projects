# Putting it all together, a multiclass problem!

# create a toy dataset
#1. importing dependancies
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd


import matplotlib.pyplot as plt
import torch
from torch import nn

import requests
from pathlib import Path
# download helper function from Learn pytorch repo
if Path("helper_functions.py").is_file():
  print("EXIST!, skipping download...")
else:
  print("NO! Downloading...")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


def accuracy_fn(y_true,y_pred):

  correct = torch.eq(y_true,y_pred).sum().item()
  acc = ((correct / len(y_pred)) * 100)
  return acc

# set the hyperparameters for data creation
NUMCLASSES = 4
NUMFEATURES = 2
RANDOMSEED = 50

#1. Create multiclass data(blobs)
XBlob, yBlob = make_blobs(n_samples=1000,n_features=NUMFEATURES,centers=NUMCLASSES, cluster_std=1.5, random_state=RANDOMSEED)

# 2. turn data into tensors
XBlob = torch.from_numpy(XBlob).type(torch.float)
yBlob = torch.from_numpy(yBlob).type(torch.float)

# 3. split into train and test sets

XTrain, XTest, yTrain, yTest = train_test_split(XBlob,yBlob,test_size=0.2,random_state=RANDOMSEED)


# 4. plot data
plt.figure(figsize=(10,7))
plt.scatter(XBlob[:,0],XBlob[:,1],c=yBlob,cmap=plt.cm.RdYlBu);

# Now this, produces a very nice plot of our data, spread into 4 colors, yellow, red, dark blue and light blue!


#5. Create model architecture
class BlobModel(nn.Module):
  def __init__(self,input_features,output_features,hidden_units=8):
    """ Initialize a multiclass classification model:
    Args:
      input features(int) = number of input features for the model
      output features(int) = same as input
      hidden units(int) = number of neurons in layers

    Returns:
      Logit for classyfying
      """
    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_features,out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units,out_features=hidden_units),
        nn.ReLU(),
        nn.Linear(in_features=hidden_units,out_features=output_features)
    )
  def forward(self,x):
    return self.linear_layer_stack(x)

# create model instance

model4=BlobModel(input_features=2,output_features=4,hidden_units=8)


# 6. Pick Loss and optim functions

# for loss with multiclass classification, we typically pick Cross Entropy Loss(Unlike for binary class, with Binary cross entropy)


loss_fn = torch.nn.CrossEntropyLoss()

optim_fn = torch.optim.SGD(params=model4.parameters(),lr=0.005)

model4.eval()
with torch.inference_mode():
  y_logits = model4(XTest)
# in order to train them, we need to go from logits ->  to probabilities -> labels
# so we need an output activation function -> softMax(common for multclass)

y_pred_probs = torch.softmax(y_logits,dim=1)

#torch.sum(y_pred_probs[0]) all probs sum up to 1
yPreds = torch.argmax(y_pred_probs,dim=1) # this outputs the class(index 3) the model thinks the first sampple(vector) is
# we turn the preds into labels
# 7 Build a training loop
torch.manual_seed(50)

epochs = 20000

for epoch in range(epochs):
  ## train
  model4.train()
  yLogits =  model4(XTrain)
  yPred =  torch.softmax(yLogits,1).argmax(dim=1) # at once turn logits - labels

  loss = loss_fn(yLogits,yTrain.type(torch.LongTensor))
  acc = accuracy_fn(y_true=yTrain,y_pred=yPred)

  optim_fn.zero_grad()
  loss.backward()
  optim_fn.step()


  # test
  model4.eval()
  with torch.inference_mode():
    testLogits = model4(XTest)
    testPreds = torch.softmax(testLogits,1).argmax(dim=1)

    testLoss = loss_fn(testLogits,yTest.type(torch.LongTensor))

    test_acc = accuracy_fn(y_true=yTest,y_pred=testPreds)

    if(epoch % 1000 == 0):
      print(f"Epoch: {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f} | Test Loss: {testLoss:.4f} | Test Acc: {test_acc:.2f}")

