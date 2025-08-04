""" app.py : This module contains a simple calculator class to performs arithmetic operations."""
class Calculator:
    """A simple calculator class that performs basic arithmetic operations."""
    def __init__(self):
        """Initializes the calculator with a result of zero."""
        self.result = 0
    def add(self, value):
        """Adds a value to the current result."""
        self.result += value
        return self.result
    def subtract(self, value):
        """Subtracts a value from the current result."""
        self.result -= value
        return self.result
    def multiply(self, value):
        """Multiplies the current result by a value."""
        self.result *= value
        return self.result
    def divide(self, value):
        """Divides the current result by a value."""
        if value == 0:
            raise ValueError("Cannot divide by zero.")
        self.result /= value
        return self.result
    def reset(self):
        """Resets the current result to zero."""
        self.result = 0
        return self.result
    def get_result(self):
        """Returns the current result."""
        return self.result
def main():
    """Main function to demonstrate the Calculator class."""
    calc = Calculator()
    print("Initial result:", calc.get_result())
    print("Adding 10:", calc.add(10))
    print("Subtracting 5:", calc.subtract(5))
    print("Multiplying by 2:", calc.multiply(2))
    print("Dividing by 3:", calc.divide(3))
    print("Current result:", calc.get_result())
    print("Resetting calculator.")
    calc.reset()
    print("Result after reset:", calc.get_result())
if __name__ == "__main__":
    main()
