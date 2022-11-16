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


### week 12 
```bash 
jena_climate_2009_2016.csv file data analysis in week 12
```
https://github.com/Alexdatser/dosronbek-AI-application-/blob/main/week%2012/week_12.ipynb



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


# Activity of week 12

## Deep learning for timeseries 
## Different kinds of timeseries tasks
## A temperature-forecasting example

```bash
!wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip 
!unzip jena_climate_2009_2016.csv.zip 
```
output:
--2022-11-15 09:37:15--  https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.63.0
Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.63.0|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 13565642 (13M) [application/zip]
Saving to: ‘jena_climate_2009_2016.csv.zip.1’

jena_climate_2009_2 100%[===================>]  12.94M  41.1MB/s    in 0.3s    

2022-11-15 09:37:16 (41.1 MB/s) - ‘jena_climate_2009_2016.csv.zip.1’ saved [13565642/13565642]

Archive:  jena_climate_2009_2016.csv.zip
replace jena_climate_2009_2016.csv? [y]es, [n]o, [A]ll, [N]one, [r]ename: y
  inflating: jena_climate_2009_2016.csv  
replace __MACOSX/._jena_climate_2009_2016.csv? [y]es, [n]o, [A]ll, [N]one, [r]ename: y
  inflating: __MACOSX/._jena_climate_2009_2016.csv  

### Inspecting the data of the Jena weather dataset

```bash
import os
fname = os.path.join("jena_climate_2009_2016.csv")
with open(fname) as f: 
  data = f.read()

lines = data.split("\n") 
header = lines[0].split(",") 
lines = lines[1:] 
print(header) 
print(len(lines)) 
```
output :
import os
fname = os.path.join("jena_climate_2009_2016.csv")
with open(fname) as f: 
  data = f.read()

lines = data.split("\n") 
header = lines[0].split(",") 
lines = lines[1:] 
print(header) 
print(len(lines)) 

### Parsing the data 
``` bash 
import numpy as np
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]
```

### Plotting the temperature timeseries

```bash
from matplotlib import pyplot as plt
plt.plot(range(len(temperature)), temperature)
```
output: 

[<matplotlib.lines.Line2D at 0x7ff82f0842d0>]
![image](https://user-images.githubusercontent.com/81208782/202094230-50dd7714-0208-4326-a8d4-b1d684510d5a.png)

### Plotting the first 10 days of the temperature timeseries

```bash
plt.plot(range(1440), temperature[:1440])
```
output:
[<matplotlib.lines.Line2D at 0x7ff82eb9add0>]
![image](https://user-images.githubusercontent.com/81208782/202094465-2717a17b-a1e1-4b2e-bf2c-568aee0bb323.png)

### Computing the number of samples we'll use for each data split
```bash
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)
```
output:

num_train_samples = int(0.5 * len(raw_data)) 
num_val_samples = int(0.25 * len(raw_data)) 
num_test_samples = len(raw_data) - num_train_samples - num_val_samples 
print("num_train_samples:", num_train_samples) 
print("num_val_samples:", num_val_samples) 
print("num_test_samples:", num_test_samples) 

## Preparing the data
### Normalizing the data
``` bash
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
```

```bash
import numpy as np
from tensorflow import keras
int_sequence = np.arange(10)
dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=int_sequence[:-3],
    targets=int_sequence[3:],
    sequence_length=3,
    batch_size=2,
)

for inputs, targets in dummy_dataset:
    for i in range(inputs.shape[0]):
        print([int(x) for x in inputs[i]], int(targets[i]))
```
output:
[0, 1, 2] 3
[1, 2, 3] 4
[2, 3, 4] 5
[3, 4, 5] 6
[4, 5, 6] 7

### Instantiating datasets for training, validation, and testing
```bash
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)
```
### Inspecting the output of one of our datasets
```bash 
for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break
```
output: 
samples shape: (256, 120, 14)
targets shape: (256,)

## A common-sense, non-machine-learning baseline
### Computing the common-sense baseline MAE
```bash 
def evaluate_naive_method(dataset):
    total_abs_err = 0.
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")
```
output:
Validation MAE: 2.44
Test MAE: 2.62

## Let's try a basic machine-learning model
### Training and evaluating a densely connected model

```bash 
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

```
output: 

Epoch 1/10
819/819 [==============================] - 56s 67ms/step - loss: 13.2852 - mae: 2.7964 - val_loss: 11.4239 - val_mae: 2.6843
Epoch 2/10
819/819 [==============================] - 62s 76ms/step - loss: 8.5516 - mae: 2.2938 - val_loss: 9.3283 - val_mae: 2.4119
Epoch 3/10
819/819 [==============================] - 64s 78ms/step - loss: 7.8400 - mae: 2.1963 - val_loss: 9.4000 - val_mae: 2.4274
Epoch 4/10
819/819 [==============================] - 61s 74ms/step - loss: 7.4414 - mae: 2.1393 - val_loss: 9.8357 - val_mae: 2.4673
Epoch 5/10
819/819 [==============================] - 56s 68ms/step - loss: 7.1592 - mae: 2.1007 - val_loss: 9.7155 - val_mae: 2.4545
Epoch 6/10
819/819 [==============================] - 55s 67ms/step - loss: 6.9160 - mae: 2.0658 - val_loss: 9.2583 - val_mae: 2.4066
Epoch 7/10
819/819 [==============================] - 57s 69ms/step - loss: 6.7467 - mae: 2.0419 - val_loss: 9.5322 - val_mae: 2.4354
Epoch 8/10
819/819 [==============================] - 55s 67ms/step - loss: 6.5934 - mae: 2.0197 - val_loss: 9.3888 - val_mae: 2.4274
Epoch 9/10
819/819 [==============================] - 59s 72ms/step - loss: 6.4723 - mae: 2.0020 - val_loss: 9.5106 - val_mae: 2.4360
Epoch 10/10
819/819 [==============================] - 59s 72ms/step - loss: 6.3713 - mae: 1.9877 - val_loss: 10.2553 - val_mae: 2.5206
406/406 [==============================] - 19s 45ms/step - loss: 9.9563 - mae: 2.4635
Test MAE: 2.46

### Plotting results

```bash
import matplotlib.pyplot as plt
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()
```
output: 
![image](https://user-images.githubusercontent.com/81208782/202095805-75d21b41-baa3-4799-821d-eb64fa02d1ff.png)

## Let's try a 1D convolutional model

```bash 
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 12, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 6, activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_conv.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model("jena_conv.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
```
output: 

Epoch 1/10
819/819 [==============================] - 97s 116ms/step - loss: 19.9688 - mae: 3.4979 - val_loss: 13.7901 - val_mae: 2.9136
Epoch 2/10
819/819 [==============================] - 89s 108ms/step - loss: 14.6891 - mae: 3.0324 - val_loss: 13.1053 - val_mae: 2.8377
Epoch 3/10
819/819 [==============================] - 91s 111ms/step - loss: 13.3938 - mae: 2.8865 - val_loss: 13.5212 - val_mae: 2.8809
Epoch 4/10
819/819 [==============================] - 90s 110ms/step - loss: 12.5220 - mae: 2.7866 - val_loss: 13.5503 - val_mae: 2.8933
Epoch 5/10
819/819 [==============================] - 91s 110ms/step - loss: 11.9395 - mae: 2.7206 - val_loss: 12.7941 - val_mae: 2.8048
Epoch 6/10
819/819 [==============================] - 91s 111ms/step - loss: 11.4450 - mae: 2.6681 - val_loss: 12.8273 - val_mae: 2.8024
Epoch 7/10
819/819 [==============================] - 90s 110ms/step - loss: 11.0313 - mae: 2.6214 - val_loss: 13.6662 - val_mae: 2.9099
Epoch 8/10
819/819 [==============================] - 91s 110ms/step - loss: 10.6990 - mae: 2.5833 - val_loss: 12.8404 - val_mae: 2.8049
Epoch 9/10
819/819 [==============================] - 89s 108ms/step - loss: 10.4212 - mae: 2.5520 - val_loss: 11.8845 - val_mae: 2.6929
Epoch 10/10
819/819 [==============================] - 90s 110ms/step - loss: 10.1486 - mae: 2.5206 - val_loss: 12.3367 - val_mae: 2.7411
406/406 [==============================] - 21s 51ms/step - loss: 13.1846 - mae: 2.8528
Test MAE: 2.85

## A first recurrent baseline
### A simple LSTM-based model
```bash 
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)

model = keras.models.load_model("jena_lstm.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
```
output: 

Epoch 1/10
819/819 [==============================] - 120s 143ms/step - loss: 42.9147 - mae: 4.7316 - val_loss: 12.1382 - val_mae: 2.6353
Epoch 2/10
819/819 [==============================] - 111s 135ms/step - loss: 9.8992 - mae: 2.4342 - val_loss: 8.4639 - val_mae: 2.2629
Epoch 3/10
819/819 [==============================] - 108s 131ms/step - loss: 8.5858 - mae: 2.2793 - val_loss: 8.2802 - val_mae: 2.2340
Epoch 4/10
819/819 [==============================] - 107s 130ms/step - loss: 8.1967 - mae: 2.2271 - val_loss: 8.1625 - val_mae: 2.2219
Epoch 5/10
819/819 [==============================] - 107s 130ms/step - loss: 7.9646 - mae: 2.1965 - val_loss: 8.1964 - val_mae: 2.2261
Epoch 6/10
819/819 [==============================] - 103s 126ms/step - loss: 7.7436 - mae: 2.1654 - val_loss: 8.4642 - val_mae: 2.2551
Epoch 7/10
819/819 [==============================] - 104s 126ms/step - loss: 7.5612 - mae: 2.1380 - val_loss: 8.4985 - val_mae: 2.2414
Epoch 8/10
819/819 [==============================] - 107s 131ms/step - loss: 7.4615 - mae: 2.1204 - val_loss: 8.8779 - val_mae: 2.2838
Epoch 9/10
819/819 [==============================] - 104s 127ms/step - loss: 7.3788 - mae: 2.1078 - val_loss: 8.6550 - val_mae: 2.2667
Epoch 10/10
819/819 [==============================] - 108s 132ms/step - loss: 7.2675 - mae: 2.0908 - val_loss: 8.6064 - val_mae: 2.2706
406/406 [==============================] - 25s 59ms/step - loss: 9.4515 - mae: 2.4115
Test MAE: 2.41

## Understanding recurrent neural networks
### NumPy implementation of a simple RNN

```bash
import numpy as np
timesteps = 100
input_features = 32
output_features = 64
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)
```

## A recurrent layer in Keras
### An RNN layer that can process sequences of any length
```bash
num_features = 14
inputs = keras.Input(shape=(None, num_features))
outputs = layers.SimpleRNN(16)(inputs)
```
### An RNN layer that returns only its last output step

```bash
num_features = 14
steps = 120
inputs = keras.Input(shape=(steps, num_features))
outputs = layers.SimpleRNN(16, return_sequences=False)(inputs)
print(outputs.shape)
```
### An RNN layer that returns its full output sequence
```bash
num_features = 14
steps = 120
inputs = keras.Input(shape=(steps, num_features))
outputs = layers.SimpleRNN(16, return_sequences=True)(inputs)
print(outputs.shape)
```
### Stacking RNN layers
```bash
inputs = keras.Input(shape=(steps, num_features))
x = layers.SimpleRNN(16, return_sequences=True)(inputs)
x = layers.SimpleRNN(16, return_sequences=True)(x)
outputs = layers.SimpleRNN(16)(x)
```
# Advanced use of recurrent neural networks
## Using recurrent dropout to fight overfitting
### Training and evaluating a dropout-regularized LSTM

```bash
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(32, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_lstm_dropout.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
          
         callbacks=callbacks)
```
```bash
inputs = keras.Input(shape=(sequence_length, num_features))
x = layers.LSTM(32, recurrent_dropout=0.2, unroll=True)(inputs)
```
## Stacking recurrent layers
### Training and evaluating a dropout-regularized, stacked GRU model
``` bash
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = layers.GRU(32, recurrent_dropout=0.5)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_stacked_gru_dropout.keras",
                                    save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("jena_stacked_gru_dropout.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
```

## Using bidirectional RNNs
### Training and evaluating a bidirectional LSTM

```bash
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Bidirectional(layers.LSTM(16))(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset)
```
