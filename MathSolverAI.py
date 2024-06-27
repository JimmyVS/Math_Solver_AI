# Math Solver AI

# Description:
# This script builds, trains, evaluates, and tests a neural network for simple arithmetic operations.
# It loads data from a CSV file containing arithmetic problems, preprocesses it by scaling the features,
# and splits it into training and testing sets.
# A neural network with three fully connected layers is defined and trained using the Adam optimizer and MSE loss function.
# The training process includes saving and loading checkpoints to resume training later.
# The model can be evaluated on a test set to check its accuracy, and it also provides a user interface for testing
# the model with custom inputs, showing both the predicted and correct results.

# License:
# MIT License. Made by JimmyVS. Please do not claim as yours.

# Import PyTorch for the creation and the training of the neural network.
import torch
import torch.nn as nn
import torch.optim as optim

# Import pandas for file handling.
import pandas as pd
# Import pyplot for the creation of graphs.
import matplotlib.pyplot as plt

# Import sklearn libraries for spliting dataset.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import time and os libraries for extra features.
import time
import os

correctResult = 0

# Load and preprocess the data.
data = pd.read_csv('operations.csv')
X = data[['Number A', 'Operation', 'Number B']].values
y = data['Solution'].values

# Scale the input features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors.
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the neural network model.
class Model(nn.Module):
    def __init__(self, input=3, h1=64, h2=32, output=1):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output)
    
    # Define the flow of the neural network.
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Initialize the model, loss function, and optimizer.
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training variables.
num_epochs = 1000
batch_size = 32
losses = []

# Function to save current progress of AI model.
def save_progress(model, optimizer, epoch, losses, filename="progress.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, filename)
    print(f"\nProgress saved to {filename}")

# Function to load previous progress of AI model.
def load_progress(filename="progress.pth"):
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        print(f"\nProgress loaded from {filename}")
        return start_epoch, losses
    else:
        print(f"\nNo valid checkpoint found at {filename}")
        return 0, [], []

# Train the AI model with the X and y train.
def TrainModel():
    # Try to load progress.
    try:
        load_progress()
    except:
        print("")
    
    # Asking for a number of total epochs.
    print("\nSelect a number of epochs for training.")
    num_epochs = input("Write an int number (Example: 1000): ")
    int(num_epochs)
    
    print("")
    print("*TRAINING PROCESS*:")
    
    beforeTime = time.time()
    for epoch in range(int(num_epochs)):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Forward pass.
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Record the loss for plotting.
        losses.append(loss.item())
        accuracy = 100 - loss

        # Print each result every 50 epochs.
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}%, Accuracy: {accuracy:.4f}%')
    
    # Calculating time passed.
    finalTime = time.time()
    timePassed = finalTime - beforeTime
    timePassed = int(timePassed)
    
    print("\n*TOTAL TIME*: " + str(timePassed) + " seconds.")
    
    # Saving current progress.
    save_progress(model, optimizer, num_epochs, losses)

# Creating a graph of the results.
def Graph():
    # Plot the training loss.
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# Function to test the AI model over 1000 tries.
def EvaluateModel():
    # Try to save progress.
    try:
        load_progress()
    except:
        print("")
        
    model.eval()  # Set the model to evaluation mode.
    
    # Asking for number of margin when evaluating AI model.
    print("\nChoose a margin for the AI results (Example: 0.5): ")
    margin = input("Write a float number(ex. 0.5): ")
    float(margin)
    
    with torch.no_grad():
        y_pred = model(X_test)
        loss = criterion(y_pred, y_test)
        
        correct = 0
        wrong = 0
        
        print("")
        
        # Print predictions and actual values for manual inspection.
        for i in range(len(y_test)):
            try:
                if loss < 0.5:
                    print(f'{i+1}) Prediction: {y_pred[i].item():.4f}, Actual: {y_test[i].item():.4f} | Status: Correct')
                    correct += 1
                else:
                    print(f'{i+1}) Prediction: {y_pred[i].item():.4f}, Actual: {y_test[i].item():.4f} | Status: Wrong')
                    wrong += 1
            except:
                print(f"An error occured while evaluating.")
                print("Solution: Set margin value to a float number.")
        
        total = correct + wrong 
        accuracy = correct / total * 100

        # Printing results.
        print(f"\n*RESULTS*: Correct: {correct}, Mistakes: {wrong}, Accuracy: {accuracy}, Total: {total}")
        print("WARNING: AI predictions are cannot be exactly correct. There is a loss ~0.5.")
        print("Please train AI model, before evaluating.")

# User interface for testing.
def predict_operation(num_a, operation, num_b):
    load_progress()
    
    # Map operation to numeric value.
    op_map = {'+': 1, '-': 2, '*': 3, '/': 4}
    op_num = op_map.get(operation, 0)
    
    # Prepare input.
    input_data = torch.FloatTensor([[num_a, op_num, num_b]])
    scaled_input = torch.FloatTensor(scaler.transform(input_data))
    
    # Make prediction.
    model.eval()
    with torch.no_grad():
        prediction = model(scaled_input)
    
    return prediction.item()

# Test the model with user input.
while True:
    print("\nSelect an option: ")
    print("1. Train Model   2. Evaluate Model   3. Delete Progress  4. Exit")
    
    choice = input('Select 1/2/3: ')
    if choice == "1":
        print("\nSelect an option: ")
        print("1. Train with graph  2. Train without graph  3. Exit")
        
        choice == input('Select 1/2/3: ')
        if choice == "1":
            TrainModel()
            Graph()
        elif choice == "2":
            TrainModel()
        elif choice == "3":
            # Saving current progress and exiting.
            save_progress(model, optimizer, num_epochs, losses)
            break
        else:
            print("Please select only 1, 2 or 3.")
    elif choice == "2":
        print("\nSelect an option: ")
        print("1. Evaluate Model    2. Test Model with your input   3. Exit")
        
        choice = input('Select 1/2/3: ')
        if choice == "1":
            EvaluateModel()
        elif choice == "2":     
            try:
                print("")
                
                # Asking user for an equation.
                num_a = float(input("Enter the first number: "))
                operation = input("Enter the operation (+, -, *, /): ")
                num_b = float(input("Enter the second number: "))
                
                # Making a prediction.
                result = predict_operation(num_a, operation, num_b)
                
                # Calculating a correct result.
                if operation == "+":
                    correctResult = num_a + num_b
                elif operation == "-":
                    correctResult = num_a - num_b
                elif operation == "*":
                    correctResult = num_a * num_b
                elif operation == "/":
                    correctResult = num_a / num_b
                
                # Printing results.
                print(f"Predicted result: {result:.2f}, Correct Result: {correctResult:.2f}")
                
                cont = input("Do you want to continue? (y/n): ")
                if cont.lower() != 'y':
                    # Saving progress and exiting.
                    save_progress(model, optimizer, num_epochs, losses)
                    break
                
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
                
        elif choice == "3":
            # Saving progress and exiting.
            save_progress(model, optimizer, num_epochs, losses)
            break
        else:
            print("Please select only 1, 2 or 3.")
                
    elif choice == "3":
        # Deleting progress from progress.pth
        print("Progress deleted.")
        with open('progress.pth', 'w') as f:
            f.write("")
        break
    elif choice == "4":
        # Saving progress and exiting.
        save_progress(model, optimizer, num_epochs, losses)
        break
    else:
        print("Please select only 1, 2 or 3.")
