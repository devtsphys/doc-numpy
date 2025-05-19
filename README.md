# NumPy Comprehensive Reference Card

## Table of Contents
1. [Introduction](#introduction)
2. [Array Creation](#array-creation)
3. [Array Attributes and Methods](#array-attributes-and-methods)
4. [Indexing and Slicing](#indexing-and-slicing)
5. [Array Manipulation](#array-manipulation)
6. [Mathematical Operations](#mathematical-operations)
7. [Statistical Functions](#statistical-functions)
8. [Linear Algebra](#linear-algebra)
9. [Random Number Generation](#random-number-generation)
10. [Broadcasting](#broadcasting)
11. [Universal Functions (ufuncs)](#universal-functions-ufuncs)
12. [Structured Arrays](#structured-arrays)
13. [File I/O](#file-io)
14. [Advanced Techniques](#advanced-techniques)
15. [Performance Tips](#performance-tips)

## Introduction

NumPy (Numerical Python) is a powerful library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

```python
import numpy as np  # Standard import convention
```

## Array Creation

| Function | Description | Example |
|----------|-------------|---------|
| `np.array()` | Create array from list/tuple | `np.array([1, 2, 3])` |
| `np.zeros()` | Create array of zeros | `np.zeros((3, 4))` |
| `np.ones()` | Create array of ones | `np.ones((2, 3))` |
| `np.empty()` | Create uninitialized array | `np.empty((2, 3))` |
| `np.arange()` | Create range array | `np.arange(0, 10, 2)` |
| `np.linspace()` | Create evenly spaced points | `np.linspace(0, 1, 5)` |
| `np.logspace()` | Create logarithmically spaced points | `np.logspace(0, 2, 5)` |
| `np.eye()` | Create identity matrix | `np.eye(3)` |
| `np.diag()` | Create diagonal array | `np.diag([1, 2, 3])` |
| `np.fromfunction()` | Create from function | `np.fromfunction(lambda i, j: i+j, (3, 3))` |
| `np.meshgrid()` | Create coordinate matrices | `np.meshgrid([1, 2], [3, 4])` |
| `np.full()` | Create array with constant value | `np.full((2, 2), 7)` |

## Array Attributes and Methods

| Attribute/Method | Description | Example |
|------------------|-------------|---------|
| `shape` | Array dimensions | `arr.shape` |
| `ndim` | Number of dimensions | `arr.ndim` |
| `size` | Total number of elements | `arr.size` |
| `dtype` | Data type | `arr.dtype` |
| `itemsize` | Size of each element (bytes) | `arr.itemsize` |
| `nbytes` | Total size (bytes) | `arr.nbytes` |
| `T` | Transpose | `arr.T` |
| `flat` | 1D iterator | `for x in arr.flat: print(x)` |
| `flatten()` | Return flattened copy | `arr.flatten()` |
| `ravel()` | Return flattened view | `arr.ravel()` |
| `reshape()` | Change shape | `arr.reshape(2, 3)` |
| `resize()` | Change shape in-place | `arr.resize((2, 3))` |
| `astype()` | Convert data type | `arr.astype(np.float64)` |

## Indexing and Slicing

| Operation | Description | Example |
|-----------|-------------|---------|
| Basic indexing | Access element | `arr[0, 2]` |
| Slicing | Extract sub-array | `arr[0:2, 1:3]` |
| Boolean indexing | Filter by condition | `arr[arr > 5]` |
| Fancy indexing | Index with array | `arr[[0, 2], [1, 3]]` |
| Ellipsis | Use `...` to select | `arr[..., 2]` |
| `np.newaxis` | Add a dimension | `arr[:, np.newaxis]` |
| Integer array indexing | Select specific elements | `arr[[1, 2, 3], [0, 1, 0]]` |
| `np.ix_()` | Create open mesh from multiple sequences | `arr[np.ix_([1, 3], [2, 5])]` |

## Array Manipulation

| Function | Description | Example |
|----------|-------------|---------|
| `reshape()` | Change shape | `np.reshape(arr, (2, 3))` |
| `resize()` | Change shape in-place | `np.resize(arr, (5, 5))` |
| `transpose()` | Permute dimensions | `np.transpose(arr)` |
| `swapaxes()` | Swap axes | `np.swapaxes(arr, 0, 1)` |
| `flatten()` | Flatten array | `arr.flatten()` |
| `ravel()` | Flatten (view) | `np.ravel(arr)` |
| `squeeze()` | Remove axes of length 1 | `np.squeeze(arr)` |
| `expand_dims()` | Add axis | `np.expand_dims(arr, axis=1)` |
| `concatenate()` | Join arrays | `np.concatenate([a, b], axis=0)` |
| `vstack()` | Stack vertically | `np.vstack([a, b])` |
| `hstack()` | Stack horizontally | `np.hstack([a, b])` |
| `dstack()` | Stack depth-wise | `np.dstack([a, b])` |
| `column_stack()` | Stack as columns | `np.column_stack([a, b])` |
| `split()` | Split array | `np.split(arr, 3)` |
| `hsplit()` | Split horizontally | `np.hsplit(arr, 3)` |
| `vsplit()` | Split vertically | `np.vsplit(arr, 3)` |
| `tile()` | Repeat array | `np.tile(arr, (2, 3))` |
| `repeat()` | Repeat elements | `np.repeat(arr, 3)` |
| `roll()` | Roll elements | `np.roll(arr, 2, axis=0)` |
| `rot90()` | Rotate 90 degrees | `np.rot90(arr)` |
| `flip()` | Reverse elements | `np.flip(arr, axis=0)` |
| `fliplr()` | Flip left/right | `np.fliplr(arr)` |
| `flipud()` | Flip up/down | `np.flipud(arr)` |

## Mathematical Operations

| Operation/Function | Description | Example |
|--------------------|-------------|---------|
| `+, -, *, /, //, %, **` | Element-wise arithmetic | `a + b`, `a * b` |
| `+=, -=, *=, /=` | In-place operations | `a += b` |
| `np.add()` | Addition | `np.add(a, b)` |
| `np.subtract()` | Subtraction | `np.subtract(a, b)` |
| `np.multiply()` | Multiplication | `np.multiply(a, b)` |
| `np.divide()` | Division | `np.divide(a, b)` |
| `np.floor_divide()` | Integer division | `np.floor_divide(a, b)` |
| `np.power()` | Power | `np.power(a, 2)` |
| `np.sqrt()` | Square root | `np.sqrt(arr)` |
| `np.exp()` | Exponential | `np.exp(arr)` |
| `np.log()`, `np.log10()` | Logarithm | `np.log(arr)` |
| `np.sin()`, `np.cos()` | Trigonometric | `np.sin(arr)` |
| `np.arcsin()`, `np.arccos()` | Inverse trig | `np.arcsin(arr)` |
| `np.degrees()` | Radians to degrees | `np.degrees(arr)` |
| `np.radians()` | Degrees to radians | `np.radians(arr)` |
| `np.clip()` | Clip values | `np.clip(arr, 0, 1)` |
| `np.round()` | Round values | `np.round(arr, 2)` |
| `np.floor()` | Floor values | `np.floor(arr)` |
| `np.ceil()` | Ceiling values | `np.ceil(arr)` |
| `np.abs()` | Absolute value | `np.abs(arr)` |
| `np.sign()` | Sign function | `np.sign(arr)` |

## Statistical Functions

| Function | Description | Example |
|----------|-------------|---------|
| `np.min()` | Minimum | `np.min(arr)` |
| `np.max()` | Maximum | `np.max(arr)` |
| `np.argmin()` | Index of minimum | `np.argmin(arr)` |
| `np.argmax()` | Index of maximum | `np.argmax(arr)` |
| `np.sum()` | Sum | `np.sum(arr, axis=0)` |
| `np.prod()` | Product | `np.prod(arr)` |
| `np.mean()` | Mean | `np.mean(arr)` |
| `np.std()` | Standard deviation | `np.std(arr)` |
| `np.var()` | Variance | `np.var(arr)` |
| `np.median()` | Median | `np.median(arr)` |
| `np.percentile()` | Percentile | `np.percentile(arr, 75)` |
| `np.quantile()` | Quantile | `np.quantile(arr, 0.75)` |
| `np.average()` | Weighted average | `np.average(arr, weights=w)` |
| `np.cov()` | Covariance | `np.cov(x, y)` |
| `np.corrcoef()` | Correlation coefficient | `np.corrcoef(x, y)` |
| `np.histogram()` | Histogram | `np.histogram(arr, bins=5)` |
| `np.bincount()` | Count occurrences | `np.bincount(arr)` |
| `np.unique()` | Unique elements | `np.unique(arr)` |
| `np.count_nonzero()` | Count non-zeros | `np.count_nonzero(arr)` |
| `np.allclose()` | Compare arrays | `np.allclose(a, b)` |
| `np.isnan()` | Check for NaN | `np.isnan(arr)` |
| `np.isinf()` | Check for Inf | `np.isinf(arr)` |
| `np.isfinite()` | Check finite | `np.isfinite(arr)` |

## Linear Algebra

| Function | Description | Example |
|----------|-------------|---------|
| `np.dot()` | Dot product | `np.dot(a, b)` |
| `@` | Matrix multiplication | `a @ b` |
| `np.matmul()` | Matrix multiplication | `np.matmul(a, b)` |
| `np.inner()` | Inner product | `np.inner(a, b)` |
| `np.outer()` | Outer product | `np.outer(a, b)` |
| `np.tensordot()` | Tensor dot product | `np.tensordot(a, b, axes=1)` |
| `np.linalg.det()` | Determinant | `np.linalg.det(arr)` |
| `np.linalg.inv()` | Inverse | `np.linalg.inv(arr)` |
| `np.linalg.solve()` | Solve linear equations | `np.linalg.solve(a, b)` |
| `np.linalg.eig()` | Eigenvalues/vectors | `np.linalg.eig(arr)` |
| `np.linalg.eigvals()` | Eigenvalues | `np.linalg.eigvals(arr)` |
| `np.linalg.svd()` | Singular value decomp | `np.linalg.svd(arr)` |
| `np.linalg.norm()` | Norm | `np.linalg.norm(arr)` |
| `np.linalg.matrix_rank()` | Matrix rank | `np.linalg.matrix_rank(arr)` |
| `np.linalg.qr()` | QR decomposition | `np.linalg.qr(arr)` |
| `np.linalg.cholesky()` | Cholesky decomp | `np.linalg.cholesky(arr)` |
| `np.linalg.lstsq()` | Least-squares solution | `np.linalg.lstsq(a, b)` |
| `np.trace()` | Trace | `np.trace(arr)` |
| `np.vdot()` | Vector dot product | `np.vdot(a, b)` |
| `np.cross()` | Cross product | `np.cross(a, b)` |

## Random Number Generation

| Function | Description | Example |
|----------|-------------|---------|
| `np.random.seed()` | Set random seed | `np.random.seed(42)` |
| `np.random.rand()` | Uniform [0,1) | `np.random.rand(3, 2)` |
| `np.random.randn()` | Standard normal | `np.random.randn(3, 2)` |
| `np.random.randint()` | Random integers | `np.random.randint(0, 10, size=(3, 3))` |
| `np.random.random()` | Random [0,1) | `np.random.random((2, 2))` |
| `np.random.normal()` | Normal distribution | `np.random.normal(0, 1, size=(2, 2))` |
| `np.random.uniform()` | Uniform distribution | `np.random.uniform(0, 1, size=(2, 2))` |
| `np.random.binomial()` | Binomial distribution | `np.random.binomial(10, 0.5, size=(2, 2))` |
| `np.random.poisson()` | Poisson distribution | `np.random.poisson(lam=5, size=(2, 2))` |
| `np.random.exponential()` | Exponential distribution | `np.random.exponential(scale=1.0, size=(2, 2))` |
| `np.random.choice()` | Random sample | `np.random.choice([1, 2, 3], size=5)` |
| `np.random.shuffle()` | Shuffle array | `np.random.shuffle(arr)` |
| `np.random.permutation()` | Random permutation | `np.random.permutation(10)` |

## Broadcasting

Broadcasting is a powerful mechanism that allows NumPy to work with arrays of different shapes for arithmetic operations.

Rules:
1. Arrays are compatible when dimensions are equal or one of them is 1
2. Arrays are aligned at their trailing (rightmost) dimensions

Example:
```python
a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
b = np.array([10, 20, 30])            # Shape: (3,)
c = a + b                             # b is broadcast to shape (2, 3)
```

Common broadcasting patterns:
```python
# Add column vector to matrix
a = np.arange(12).reshape(4, 3)
b = np.array([1, 2, 3])
c = a + b  # b is broadcast to each row

# Add row vector to matrix
a = np.arange(12).reshape(4, 3)
b = np.array([[10], [20], [30], [40]])
c = a + b  # b is broadcast to each column
```

## Universal Functions (ufuncs)

Universal functions operate element-wise on arrays, supporting broadcasting, type casting, and other features.

| Category | Examples |
|----------|----------|
| Math operations | `add`, `subtract`, `multiply`, `divide`, `power` |
| Trigonometric | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan` |
| Bit-twiddling | `bitwise_and`, `bitwise_or`, `invert`, `left_shift` |
| Comparison | `greater`, `greater_equal`, `less`, `less_equal`, `equal` |
| Floating point | `isnan`, `isinf`, `isfinite`, `signbit` |

Methods of ufuncs:
- `reduce`: Reduces array using operation (`np.add.reduce(arr)`)
- `accumulate`: Accumulates operation results (`np.add.accumulate(arr)`)
- `outer`: Apply operation to all pairs (`np.multiply.outer(a, b)`)
- `at`: Perform operation in-place at indices (`np.add.at(arr, indices, values)`)

## Structured Arrays

Structured arrays are arrays with compound data types.

```python
# Creating structured arrays
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
arr = np.array([('Alice', 25, 55.0), ('Bob', 30, 70.5)], dtype=dt)

# Accessing fields
names = arr['name']
ages = arr['age']

# Filtering
seniors = arr[arr['age'] > 27]
```

## File I/O

| Function | Description | Example |
|----------|-------------|---------|
| `np.save()` | Save array to .npy | `np.save('array.npy', arr)` |
| `np.savez()` | Save multiple arrays | `np.savez('arrays.npz', a=arr1, b=arr2)` |
| `np.savez_compressed()` | Save compressed | `np.savez_compressed('arrays.npz', a=arr1)` |
| `np.load()` | Load .npy or .npz | `data = np.load('array.npy')` |
| `np.loadtxt()` | Load from text | `arr = np.loadtxt('data.csv', delimiter=',')` |
| `np.savetxt()` | Save to text | `np.savetxt('out.csv', arr, delimiter=',')` |
| `np.genfromtxt()` | Load with missing values | `arr = np.genfromtxt('data.csv', delimiter=',')` |
| `np.fromfile()` | Load from binary | `arr = np.fromfile('data.bin', dtype=np.float64)` |
| `np.tofile()` | Save to binary | `arr.tofile('data.bin')` |

## Advanced Techniques

### Vectorization

Vectorization involves using NumPy operations instead of loops for better performance.

```python
# Instead of:
result = []
for i in range(len(x)):
    result.append(x[i] ** 2 + y[i])

# Use:
result = x**2 + y
```

### Memory Views and Striding

Access array data with different memory layouts:

```python
# Create view with different strides
x = np.arange(16).reshape(4, 4)
y = np.lib.stride_tricks.as_strided(x, shape=(2, 2), strides=(16, 4))
```

### Masked Arrays

Handle missing or invalid data:

```python
# Create masked array
data = np.array([1, 2, np.nan, 4])
masked_data = np.ma.masked_invalid(data)

# Operations skip masked elements
mean = np.ma.mean(masked_data)
```

### Polynomial Functions

```python
# Create polynomial
p = np.polynomial.Polynomial([1, 2, 3])  # 1 + 2x + 3x²

# Evaluate
result = p(2)  # 1 + 2*2 + 3*2² = 17

# Operations
p_derivative = p.deriv()
p_integral = p.integ()
roots = p.roots()
```

### Fast Fourier Transform

```python
# Compute FFT
x = np.array([1, 2, 1, 0, 1, 2, 1, 0])
fft = np.fft.fft(x)

# Inverse FFT
ifft = np.fft.ifft(fft)

# 2D FFT
fft2 = np.fft.fft2(img)
```

### Set Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([3, 4, 5, 6])

# Set operations
intersection = np.intersect1d(a, b)  # [3, 4]
union = np.union1d(a, b)             # [1, 2, 3, 4, 5, 6]
difference = np.setdiff1d(a, b)      # [1, 2]
in_both = np.in1d(a, b)              # [False, False, True, True]
```

## Performance Tips

1. **Vectorize operations**: Avoid Python loops when possible
   ```python
   # Bad: Using loops
   for i in range(len(a)):
       c[i] = a[i] + b[i]
   
   # Good: Vectorized
   c = a + b
   ```

2. **Use views instead of copies**: When possible, use array views
   ```python
   # Creates a view
   b = a.reshape(3, 4)  
   
   # Creates a copy
   b = a.reshape(3, 4).copy()
   ```

3. **Choose appropriate data types**: Use smallest data type needed
   ```python
   # Use 16-bit integers if sufficient
   arr = np.array([1, 2, 3], dtype=np.int16)
   ```

4. **Pre-allocate arrays**: Create output arrays beforehand
   ```python
   result = np.empty_like(arr)
   ```

5. **Use compiled functions**: Choose built-in NumPy functions over custom functions
   ```python
   # Slow: Custom function
   def custom_mean(x):
       return sum(x) / len(x)
   
   # Fast: NumPy function
   result = np.mean(x)
   ```

6. **Use `where` for conditional assignments**:
   ```python
   # Instead of:
   for i in range(len(a)):
       if a[i] < 0:
           a[i] = 0
   
   # Use:
   a = np.where(a < 0, 0, a)
   ```

7. **Avoid unnecessary copies**: Use in-place operations when possible
   ```python
   # Creates new array
   b = a + 1
   
   # Modifies in-place
   a += 1
   ```

8. **Use `einsum` for complex operations**: Efficient for many linear algebra operations
   ```python
   # Matrix multiplication
   c = np.einsum('ij,jk->ik', a, b)
   
   # Batch matrix multiplication
   c = np.einsum('bij,bjk->bik', a, b)
   ```