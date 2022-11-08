### week 3
```bash 
week 3 activities of class , Installing tensorflow
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%203


### week 4
```bash 
Activities we did in week 4 ( google collab introduction with some installations)
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%204


### week5
```bash
Translow basic operations done in week 5 class
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%204


### week6 
``` bash 
Linear Regression
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%206


### week7
```bash how to access the CIFAR-10 dataset and display the image with index 100
convolutional neural network to solve an image classification problem, using the CIFAR-10 dataset 
almost same as week7-2 but the network is modified to use more complex network
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%207


### week 8
```bash 
In this week we did midterm exam and for exam i submittied all weekly assignments as usual
but I did not create a new reprosotory for midterm itself since all weekly assigmnetnts i did so far can be 
a work of midterm
```


### week 9
```bash 
Creating and training the tensorflow in week 9
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%209


### week 10
```bash 
working with torch and loading , preprocessing the image in week 10
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%2010


### week 11
```bash
In this week we worked on some dataset like Plot dataset and Plot naive prediction
with the book sales dataset exel file 
```
https://github.com/Alexdatser/dosronbek-AI-application-/tree/main/week%2011



# Linear Regression using TensorFlow

```bash
# First install the tensorFlow version 1.15.5
---------------------------------------------
pip install tensorflow==1.15.5

#import the libraries
----------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_parameter = 0.01
epochs = 300

sample_points = 50
x_train = np.linspace(0, 30, sample_points)
y_train = 6*x_train + 7*np.random.randn(sample_points)

# Noisy dataset
plt.plot(x_train, y_train, 'o')
# Noise free dataset 
plt.plot(x_train, 6*x_train)
plt.show()

Y = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(), name = 'weights')
B = tf.Variable(np.random.randn(), name = 'bias')

#Create the model for regression
prediction = W*X + B

# Cost function
cost_iteration = tf.reduce_sum((prediction-Y)**2)/(2*sample_points)

#Define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_parameter).minimize(cost_iteration)

# Initialize the variables
init = tf.global_variables_initializer()

# Start the Tensorflow session to execute the graph
---------------------------------------------------
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
      sess.run(optimizer, feed_dict = {X : x, Y : y})
    if not epoch%40:
      W1 = sess.run(W)
      B1 = sess.run(B)
      cost_iter = sess.run(cost_iteration, feed_dict = {X : x, Y : y})
      print('Epochs %f Cost %f Weight %f Bias %f' %(epoch, cost_iter, W1, B1))
  Weight = sess.run(W)
  Bias = sess.run(B)

  plt.plot(x_train, y_train, 'o')
  plt.plot(x_train,Weight*x_train+Bias)
  plt.show()
  
#output
-------
Epochs 0.000000 Cost 0.154752 Weight 5.665872 Bias 0.810562
Epochs 40.000000 Cost 0.389723 Weight 6.009754 Bias 0.670748
Epochs 80.000000 Cost 0.392550 Weight 6.015407 Bias 0.523763
Epochs 120.000000 Cost 0.395118 Weight 6.020525 Bias 0.390661
Epochs 160.000000 Cost 0.397452 Weight 6.025161 Bias 0.270135
Epochs 200.000000 Cost 0.399572 Weight 6.029358 Bias 0.160994
Epochs 240.000000 Cost 0.401493 Weight 6.033159 Bias 0.062164
Epochs 280.000000 Cost 0.403241 Weight 6.036601 Bias -0.027329
```
![image](https://user-images.githubusercontent.com/81208782/197143098-a410ae18-02ed-4b0e-9620-1fed9303af17.png)

##parameters using TensorFlow
```bash
# Create the model for regression
with tf.name_scope("Model") as scope:
  prediction = W*X + B

# Add summary to study behaviour of weights and biases with epochs
weight_histogram = tf.summary.histogram("Weights", W)
bias_histogram = tf.summary.histogram("Bias", B)

# Cost function
with tf.name_scope("Cost_function") as scope:
  cost_iteration = tf.reduce_sum((prediction-Y)**2)/(2*sample_points)

# Record the scalar summary of the cost function
cost_summary = tf.summary.scalar("Cost", cost_iteration)

#Define the optimizer
with tf.name_scope("Training") as scope:
  optimizer = tf.train.GradientDescentOptimizer(learning_parameter).minimize(cost_iteration)

# Initialize the variables
init = tf.global_variables_initializer()

#Merge all the summaries into a single operator
merged_summary = tf.summary.merge_all()

# Define the tensorflow session
with tf.Session() as sess:
  sess.run(init)
  writer = tf.summary.FileWriter('./log', sess.graph)
  for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
      sess.run(optimizer, feed_dict = {X : x, Y : y})

      # Write logs for each epochs
      summary_epochs = sess.run(merged_summary, feed_dict = {X : x, Y : y})
      writer.add_summary(summary_epochs, epoch)
    if not epoch%40:
      W1 = sess.run(W)
      B1 = sess.run(B)
      cost_iter = sess.run(cost_iteration, feed_dict = {X : x, Y : y})
      print('Epochs %f Cost %f Weight %f Bias %f' %(epoch, cost_iter, W1, B1))
  Weight = sess.run(W)
  Bias = sess.run(B)

  plt.plot(x_train, y_train, 'o')
  plt.plot(x_train,Weight*x_train+Bias)
  plt.show()
  
#output
-------
Epochs 0.000000 Cost 1.294635 Weight 5.715763 Bias 0.545876
Epochs 40.000000 Cost 0.066145 Weight 6.013482 Bias 0.420641
Epochs 80.000000 Cost 0.065112 Weight 6.018522 Bias 0.289603
Epochs 120.000000 Cost 0.064184 Weight 6.023086 Bias 0.170943
Epochs 160.000000 Cost 0.063350 Weight 6.027218 Bias 0.063494
Epochs 200.000000 Cost 0.062598 Weight 6.030960 Bias -0.033805
Epochs 240.000000 Cost 0.061923 Weight 6.034348 Bias -0.121911
Epochs 280.000000 Cost 0.061313 Weight 6.037417 Bias -0.201694
```
![image](https://user-images.githubusercontent.com/81208782/197143819-681573ce-448c-4587-bfe2-95b0136f2a37.png)

##parameters using TensorFlow
```bash
# Create the model for regression
with tf.name_scope("Model") as scope:
  prediction = W*X + B

# Add summary to study behaviour of weights and biases with epochs
weight_histogram = tf.summary.histogram("Weights", W)
bias_histogram = tf.summary.histogram("Bias", B)

# Cost function
with tf.name_scope("Cost_function") as scope:
  cost_iteration = tf.reduce_sum((prediction-Y)**2)/(2*sample_points)

# Record the scalar summary of the cost function
cost_summary = tf.summary.scalar("Cost", cost_iteration)

#Define the optimizer
with tf.name_scope("Training") as scope:
  optimizer = tf.train.GradientDescentOptimizer(learning_parameter).minimize(cost_iteration)

# Initialize the variables
init = tf.global_variables_initializer()

#Merge all the summaries into a single operator
merged_summary = tf.summary.merge_all()

# Define the tensorflow session
with tf.Session() as sess:
  sess.run(init)
  writer = tf.summary.FileWriter('./log', sess.graph)
  for epoch in range(epochs):
    for x, y in zip(x_train, y_train):
      sess.run(optimizer, feed_dict = {X : x, Y : y})

      # Write logs for each epochs
      summary_epochs = sess.run(merged_summary, feed_dict = {X : x, Y : y})
      writer.add_summary(summary_epochs, epoch)
    if not epoch%40:
      W1 = sess.run(W)
      B1 = sess.run(B)
      cost_iter = sess.run(cost_iteration, feed_dict = {X : x, Y : y})
      print('Epochs %f Cost %f Weight %f Bias %f' %(epoch, cost_iter, W1, B1))
  Weight = sess.run(W)
  Bias = sess.run(B)

  plt.plot(x_train, y_train, 'o')
  plt.plot(x_train,Weight*x_train+Bias)
  plt.show()
  
#output
-------
Epochs 0.000000 Cost 1.294635 Weight 5.715763 Bias 0.545876
Epochs 40.000000 Cost 0.066145 Weight 6.013482 Bias 0.420641
Epochs 80.000000 Cost 0.065112 Weight 6.018522 Bias 0.289603
Epochs 120.000000 Cost 0.064184 Weight 6.023086 Bias 0.170943
Epochs 160.000000 Cost 0.063350 Weight 6.027218 Bias 0.063494
Epochs 200.000000 Cost 0.062598 Weight 6.030960 Bias -0.033805
Epochs 240.000000 Cost 0.061923 Weight 6.034348 Bias -0.121911
Epochs 280.000000 Cost 0.061313 Weight 6.037417 Bias -0.201694
```
![image](https://user-images.githubusercontent.com/81208782/197144257-85947e63-d43b-4c4b-b5c7-b00073ce826f.png)

![image](https://user-images.githubusercontent.com/81208782/197144390-f43e092e-1e4f-481a-a6e1-4763a389805a.png)

### week 10 activity
```bash
import torch
import torchvision
from torchvision import transforms
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

```bash 
# Load and preprocess image.
image=Image.open(r"RUS.jpg")
preprocess=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
input_tensor = preprocess(image)

# Convert to 4-dimensional tensor.
inputs=input_tensor.unsqueeze(0)
```

``` bash
# Load the pre—trained model. 
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1) 
# model = torchvision.models.resnet50(weights=torchvision.models.ResNet5O_Weights.IMAGENET1K_V1) 
model.eval() 
# Transfer model to GPU. 
model.to(device) 
# Do prediction. 
inputs = inputs.to(device) 
with torch.no_grad(): 
  outputs = model(inputs) 
# Convert to probabilities, since final SoftMax activation is not in pretrained model. 
probabilities = torch.nn.functional.softmax(outputs[0], dim=0) 
# Print class ID for top 5 predictions. 
_, indices = torch.sort(probabilities, descending=True) 
for i in range(0, 5): 
  print('ImageNet class:', indices[i].item(), ', probability = %4.3f' % probabilities[indices[i]].item()) 
# Show image. 
image.show()
```
Here is the output
```bash
Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
  0%|          | 0.00/97.8M [00:00, ?B/s]
ImageNet class: 617 , probability = 0.154
ImageNet class: 834 , probability = 0.088
ImageNet class: 652 , probability = 0.084
ImageNet class: 906 , probability = 0.062
ImageNet class: 457 , probability = 0.045
```
### Activity that has been done in week 11
``` bash
import requests
# Save datagenerators as file to colab working directory
# If you are using GitHub, make sure you get the "Raw" version of the code
url = 'https://raw.githubusercontent.com/NVDLI/LDL/main/pt_framework/utilities.py'
r = requests.get(url)
# make sure your filename is the same as how you want to import
with open('utilities.py','w') as f:
  f.write(r.text)
```

``` bash 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utilities import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16

TRAIN_TEST_SPLIT = 0.8
MIN = 12
FILE_NAME = 'book_store_sales.csv'

def readfile(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    next(file)
    data = []
    for line in (file):
        values = line.split(',')
        data.append(float(values[1]))
    file.close()
    return np.array(data, dtype=np.float32)

# Read data and split up into train and test data.
sales = readfile(FILE_NAME)
months = len(sales)
split = int(months * TRAIN_TEST_SPLIT)
train_sales = sales[0:split]
test_sales = sales[split:]
```

```bash
# Plot dataset
x = range(len(sales))
plt.plot(x, sales, 'r-', label='book sales')
plt.title('Book store sales')
plt.axis([0, 339, 0.0, 3000.0])
plt.xlabel('Months')
plt.ylabel('Sales (millions $)')
plt.legend()
plt.show()
```
Here is the output data

![image](https://user-images.githubusercontent.com/81208782/200541502-b6f1cdf4-dc97-452a-bb68-bd7b8909349a.png)

```bash
# Plot naive prediction
test_output = test_sales[MIN:]
naive_prediction = test_sales[MIN-1:-1]
x = range(len(test_output))
plt.plot(x, test_output, 'g-', label='test_output')
plt.plot(x, naive_prediction, 'm-', label='naive prediction')
plt.title('Book store sales')
plt.axis([0, len(test_output), 0.0, 3000.0])
plt.xlabel('months')
plt.ylabel('Monthly book store sales')
plt.legend()
plt.show()
```
Here is the output of this code

![image](https://user-images.githubusercontent.com/81208782/200541656-377a0999-7caf-449f-89ba-1b81167fbd78.png)

```bash
# Standardize train and test data.
# Use only training seasons to compute mean and stddev.
mean = np.mean(train_sales)
stddev = np.std(train_sales)
train_sales_std = (train_sales - mean)/stddev
test_sales_std = (test_sales - mean)/stddev
```

```bash
# Create train examples.
train_months = len(train_sales)
train_X = np.zeros((train_months-MIN, train_months-1, 1), dtype=np.float32)
train_y = np.zeros((train_months-MIN, 1), dtype=np.float32)
for i in range(0, train_months-MIN):
    train_X[i, -(i+MIN):, 0] = train_sales_std[0:i+MIN]
    train_y[i, 0] = train_sales_std[i+MIN]

# Create test examples.
test_months = len(test_sales)
test_X = np.zeros((test_months-MIN, test_months-1, 1), dtype=np.float32)
test_y = np.zeros((test_months-MIN, 1), dtype=np.float32)
for i in range(0, test_months-MIN):
    test_X[i, -(i+MIN):, 0] = test_sales_std[0:i+MIN]
    test_y[i, 0] = test_sales_std[i+MIN]

# Create Dataset objects.
trainset = TensorDataset(torch.from_numpy(train_X).clone(), torch.from_numpy(train_y))
testset = TensorDataset(torch.from_numpy(test_X).clone(), torch.from_numpy(test_y))
```

```bash
# Custom layer that retrieves only last time step from RNN output.
class LastTimestep(nn.Module):
    def forward(self, inputs):
        return inputs[1][0]

# Create RNN model
model = nn.Sequential(
    nn.RNN(1, 128, nonlinearity='relu', batch_first=True),
    LastTimestep(),
    nn.Linear(128, 1)
)

# Loss function and optimizer.
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.MSELoss()

# Train model.
train_model(model, device, EPOCHS, BATCH_SIZE, trainset, testset,
            optimizer, loss_function, 'mae')
```
Output of this code

```bash
Epoch 1/100 loss: 0.9439 - mae: 0.7205 - val_loss: 1.1952 - val_mae: 0.8513
Epoch 2/100 loss: 0.8227 - mae: 0.6574 - val_loss: 1.0075 - val_mae: 0.7842
Epoch 3/100 loss: 0.6335 - mae: 0.5761 - val_loss: 0.7412 - val_mae: 0.6464....


...Epoch 99/100 loss: 0.0283 - mae: 0.1248 - val_loss: 0.0548 - val_mae: 0.1368
Epoch 100/100 loss: 0.0248 - mae: 0.1153 - val_loss: 0.0567 - val_mae: 0.1423
[0.11530081007410498, 0.14234759286046028]
```

```bash
# Create naive prediction based on standardized data.
test_output = test_sales_std[MIN:]
naive_prediction = test_sales_std[MIN-1:-1]
mean_squared_error = np.mean(np.square(naive_prediction
                                       - test_output))
mean_abs_error = np.mean(np.abs(naive_prediction
                                - test_output))
print('naive test mse: ', mean_squared_error)
print('naive test mean abs: ', mean_abs_error)
```
Output:
```bash
naive test mse:  0.47174773
naive test mean abs:  0.48024118
```

```bash
# Use trained model to predict the test data
inputs = torch.from_numpy(test_X)
inputs = inputs.to(device)
outputs = model(inputs)
predicted_test = outputs.cpu().detach().numpy()

# De-standardize output.
predicted_test = np.reshape(predicted_test,
                            (len(predicted_test)))
predicted_test = predicted_test * stddev + mean

# Plot test prediction.
x = range(len(test_sales)-MIN)
plt.plot(x, predicted_test, 'm-',
         label='predicted test_output')
plt.plot(x, test_sales[-(len(test_sales)-MIN):],
         'g-', label='actual test_output')
plt.title('Book sales')
plt.axis([0, 55, 0.0, 3000.0])
plt.xlabel('months')
plt.ylabel('Predicted book sales')
plt.legend()
plt.show()
```
Output image:
![image](https://user-images.githubusercontent.com/81208782/200542608-a1c902b5-768c-49e9-a324-7f9be94470b4.png)
