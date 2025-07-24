from numpy import *
#import matplotlib.pyplot as plt


def compute_error_for_line_given_points(b, m, points):
    # Let's initialize error
    TotalError = 0
    for i in range(0, len(points)):
        # get the x value
        x = points[:, 0]
        # get the y value
        y = points[:, 1]
        # Now, get the difference, square it, and add it to the total error
        TotalError += (y - (m * x + b)) ** 2
    
    # Return the average error
    return TotalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # Initialize b and m
    b = starting_b
    m = starting_m

    # Gradient descent algorithm
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    
    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    # Initialize gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    # Calculate the gradient
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction with respect to b and m
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    # Update the b and m values using the gradients
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]


def run():
    # Collect the data
    points = genfromtxt('C:\\Users\\itoro\\OneDrive\\Documents\\Code\\data.csv', delimiter=',')

    # Separate x and y values
    x = points[:, 0]  # All rows, first column (x-values)
    y = points[:, 1]  # All rows, second column (y-values)
    """
    # Create the plot
    plt.scatter(x, y)  # Scatter plot of the data points
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Data and Initial Line of Best Fit')
    """
    # Define the hyperparameters
    learning_rate = 0.0001  # The rate at which the model learns
    initial_b = 0  # Initial bias/y intercept
    initial_m = 1  # Initial slope
    # Remember the straight line formula? y = mx + c? Except we're using b here so it's y = mx + b
    num_iterations = 1  # Number of iterations to run the gradient descent algorithm

    # train the model
    initial_error = compute_error_for_line_given_points(initial_b, initial_m, points)
    print(f'Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {initial_error}')

    print("Running...")  # Added this line

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    final_error = compute_error_for_line_given_points(b, m, points)
    print(f'After {num_iterations} iterations b = {b}, m = {m}, error = {final_error}') # Modified the output


if __name__ == '__main__':
=======
from numpy import *
#import matplotlib.pyplot as plt


def compute_error_for_line_given_points(b, m, points):
    # Let's initialize error
    TotalError = 0
    for i in range(0, len(points)):
        # get the x value
        x = points[:, 0]
        # get the y value
        y = points[:, 1]
        # Now, get the difference, square it, and add it to the total error
        TotalError += (y - (m * x + b)) ** 2
    
    # Return the average error
    return TotalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # Initialize b and m
    b = starting_b
    m = starting_m

    # Gradient descent algorithm
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    
    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    # Initialize gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    # Calculate the gradient
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # Direction with respect to b and m
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    # Update the b and m values using the gradients
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]


def run():
    # Collect the data
    points = genfromtxt('C:\\Users\\itoro\\OneDrive\\Documents\\Code\\data.csv', delimiter=',')

    # Separate x and y values
    x = points[:, 0]  # All rows, first column (x-values)
    y = points[:, 1]  # All rows, second column (y-values)
    """
    # Create the plot
    plt.scatter(x, y)  # Scatter plot of the data points
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Data and Initial Line of Best Fit')
    """
    # Define the hyperparameters
    learning_rate = 0.0001  # The rate at which the model learns
    initial_b = 0  # Initial bias/y intercept
    initial_m = 1  # Initial slope
    # Remember the straight line formula? y = mx + c? Except we're using b here so it's y = mx + b
    num_iterations = 1  # Number of iterations to run the gradient descent algorithm

    # train the model
    initial_error = compute_error_for_line_given_points(initial_b, initial_m, points)
    print(f'Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {initial_error}')

    print("Running...")  # Added this line

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    final_error = compute_error_for_line_given_points(b, m, points)
    print(f'After {num_iterations} iterations b = {b}, m = {m}, error = {final_error}') # Modified the output


if __name__ == '__main__':
    run()