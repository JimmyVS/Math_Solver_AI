# Math Solver AI

![Figure_1](https://github.com/JimmyVS/Math_Solver_AI/assets/96888699/6ad1e19c-82cd-4eb1-a2dc-03479f9783c9)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.6.0%2B-orange.svg)

## Math Solver AI

Math Solver AI is a neural network designed to solve simple arithmetic operations. It is built using PyTorch and includes functionalities to train, evaluate, and test the model. The model can be trained on a dataset of arithmetic problems, and its performance can be evaluated on a test set. Additionally, users can input their own arithmetic problems to see the model's predictions.

## Features

- **Train the Model**: Train the neural network on a dataset of arithmetic problems.
- **Evaluate the Model**: Evaluate the model's performance on a test set.
- **Test Custom Inputs**: Input your own arithmetic problems to see the model's predictions.
- **Save and Load Checkpoints**: Save the model's progress and load it later to resume training.

## Requirements

- Python 3.6+
- PyTorch 1.6.0+
- pandas
- scikit-learn
- matplotlib

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/JimmyVS/Math_Solver_AI.git
    cd Math_Solver_AI
    ```

2. Install the required packages:
    ```bash
    pip install torch pandas scikit-learn matplotlib
    ```

3. Prepare your dataset as a CSV file with columns `Number A`, `Operation`, `Number B`, and `Solution`. An example dataset is provided below:

    | Number A | Operation | Number B | Solution |
    |----------|-----------|----------|----------|
    | 1        | +         | 1        | 2        |
    | 2        | -         | 1        | 1        |
    | 3        | *         | 2        | 6        |
    | 4        | /         | 2        | 2        |

4. Run the script:
    ```bash
    python math_solver.py
    ```

5. Follow the on-screen instructions to train, evaluate, or test the model.

## License
This repository is under a MIT license. This repository was made by JimmyVS. Please do not claim as your's.
