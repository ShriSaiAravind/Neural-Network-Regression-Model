# Ex 1 Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective is to build and train a neural network for regression using PyTorch. A dataset containing input-output pairs is preprocessed, normalized, and split into training and test sets. A neural network with fully connected layers is designed and trained using backpropagation. The goal is to predict continuous target values based on input features.

## Neural Network Model

![alt text](nn.svg)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
### Name: SHRI SAI ARAVIND R
### Register Number: 212223040197
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,10)
        self.fc2=nn.Linear(10,14)
        self.fc3=nn.Linear(14,1)
        self.relu=nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
# Write your code here
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)

# Name: HARINI R
# Register Number: 212223100010
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

<img width="170" height="580" alt="image" src="https://github.com/user-attachments/assets/1ac4a0d1-dbb0-480e-acb3-de5ba520841b" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="801" height="562" alt="image" src="https://github.com/user-attachments/assets/99eee9a3-5917-49a3-a541-83d00b681f25" />


### New Sample Data Prediction

<img width="963" height="297" alt="image" src="https://github.com/user-attachments/assets/9c74e580-69a7-4d16-9856-f4dbe8716e7f" />


## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
