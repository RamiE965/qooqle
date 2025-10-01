# Grover’s Algorithm Demo with Qiskit

This demo demonstrates the difference between a **classical search** and a **quantum search** using the **Grover’s Algorithm**.

First, we simulated finding a 'winner' value in a list using a brute force algorithm. Then we implemented Grover's algorithm in Qiskit to solve it more efficiently.

## Classical Search

We define the following:
- A list of numbers `my_list`.
- An oracle function `the_oracle()` that returns `True` if the input equals the "winner".

The brute force search goes through the list until it finds the winner.
On average, this requires N/2 calls for a list of size N.

 ## Quantum Search (Grover’s Algorithm)

 Grover’s algorithm can find the marked element in about √N calls

 Steps:
 1. Superposition
 2. Oracle
 3. Reflection
 4. Measurement