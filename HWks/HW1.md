# Prerequisites



```python
import numpy as np
```

#### Question 1 - Dot and Cross


```python
u = np.array([2, 4, -1])
v = np.array([1, -3, 5])
np.dot(u, v)
```




    -15



#### Question 2 - Matrix Multiplications


```python
A = np.array([[3, -1],
              [2, 4]])
B = np.array([[5, 2], 
              [-1, 0]])
print(np.matmul(A, B))
print(f'{np.linalg.det(A):.0f}')

```

    [[16  6]
     [ 6  4]]
    14
    

#### Question 3 - Probability


```python
print("3 / 6")
```

    3 / 6
    

#### Question 4 - Probability 


```python
print("26 / 26")
```

    26 / 26
    

#### Question 5 - Prime Numbers



```python
first_five_primes = [2, 3, 5, 7, 11]
first_five_primes
```




    [2, 3, 5, 7, 11]



#### Question 6 - Even / Odd


```python
def is_even(num: int) -> str:
    return "Odd" if num % 2 else "Even"
```

#### Question 7 - NumPy Matrices


```python
randarray = np.random.rand(3, 3)
print(randarray)
transposed = randarray.T
print(transposed)

```

    [[0.61988056 0.08312719 0.49295522]
     [0.70129885 0.24699207 0.6391238 ]
     [0.25694693 0.71828367 0.76891171]]
    [[0.61988056 0.70129885 0.25694693]
     [0.08312719 0.24699207 0.71828367]
     [0.49295522 0.6391238  0.76891171]]
    

#### Question 8 - Simulation


```python
theoretical = 1 / 6
monte = [0 for _ in range(6)]
for inx in range(100):
    carlo = np.random.randint(1, 6)
    monte[carlo] += 1

print(f"The simulated value is {monte[3] / 100:.4f}")
print(f"The theoretical value is {theoretical:.4f}")

```

    The simulated value is 0.1900
    The theoretical value is 0.1667
    
