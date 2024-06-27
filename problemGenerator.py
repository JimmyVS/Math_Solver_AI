# Operations Generator.

# Description:
# This script creates several of meathematic operations, using random library and pandas,
# to save them to a csv. It randomly chooses an operation symbol and two numbers,
# from 1 to 20. It generates 20000 lines of mathematic operations with 4 columns.

# License:
# MIT License. Made by JimmyVS. Please do not claim as yours.

# Import the necessary libraries.
import random
import pandas as pd

# Function to generate a random operation and calculate the solution.
def generate_operation():
    number_a = random.randint(1, 20)
    number_b = random.randint(1, 20)
    operation = random.randint(1, 4)
    
    if operation == 1:
        op_symbol = '+'
        solution = number_a + number_b
    elif operation == 2:
        op_symbol = '-'
        solution = number_a - number_b
    elif operation == 3:
        op_symbol = '*'
        solution = number_a * number_b
    elif operation == 4:
        op_symbol = '/'
        solution = number_a / number_b
    
    return float(number_a), float(operation), float(number_b), solution

# Generate 5000 operations.
operations = [generate_operation() for _ in range(20000)]

# Create a DataFrame.
df = pd.DataFrame(operations, columns=['Number A', 'Operation', 'Number B', 'Solution'])

# Save to a CSV file.
df.to_csv('operations.csv', index=False)

print("Operations have been saved to operations.csv")
