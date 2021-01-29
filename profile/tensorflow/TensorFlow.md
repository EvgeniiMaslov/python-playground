**Tensorflow**

* [Tensors](#tensors)
* Manipulating tensors
* Applying mathematical operations
* Split, stack, and concatenate tensors
* Building input pipelines using tf.data – the TensorFlow Dataset API
* Combining two tensors into a joint dataset
* Shuffle, batch and repeat
* [Load dataset](#load-dataset)
* Building an NN model in TensorFlow
* [Tensorflow graph](#tensorflow-graph)
* Loading input data into model
* Improving computational performance with function decoration
* Computing the gradients of the loss with respect to trainable variables
* Writing custom Keras layers
* TensorFlow estimators







## Tensorflow



### Tensors

```python
import numpy as np
import tensorflow as tf

a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]
#create tensors
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

t_ones = tf.ones((2,3))
#to get access to values
t_ones.numpy()
#creating tensor of constant
const_tensor = tf.constane([1, 5.2, np.pi], dtype=tf.float32)
```

### Manipulating tensors

```python
#change dtype
t_a_new = tf.cast(t_a, tf.int32)
#transpose
t = tf.random.uniform(shape=(3,5))
t_tr = tf.transpose(t)
#reshape
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5,6))
#removing dimensions
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4)) # (1, 2, 1, 4, 1) -> (1, 2, 4)
```

### Applying mathematical operations

```python
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2), minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5, 2), mean=0.0, stddev=1.0)
#multiply (element-wise)
t3 = tf.multiply(t1, t2).numpy()

'''To compute the mean, sum, and standard deviation along a certain axis (or axes), we
can use tf.math.reduce_mean(), tf.math.reduce_sum(), and tf.math.reduce_
std().'''

t4 = tf.math.reduce_mean(t1, axis=0)

#matrix-matrix product
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
#tf.norm()
norm_t1 = tf.norm(t1, ord=2, axis=1).numpy()
```

### Split, stack, and concatenate tensors

```python
#splits with providing number of splits(must be divisible)
t = tf.random.uniform((6,))
t_splits = tf.splits(t, num_or_size_splits=3)
[item.numpy() for item in t_splits]
#specify the sizes of the output tensors directly
t = tf.random.uniformly((5,))
t_splits = tf.splits(t, num_or_size_splits=[3, 2])
#stack and concat
A = tf.ones((3,))
B = tf.zeros((2,))
C = tf.concat([A, B], axis=0) #[1, 1, 1, 0, 0]

A = tf.ones((3,))
B = tf.zeros((3,))
C = tf.concat([A, B], axis=1) #[[1, 1, 1], [0, 0, 0]]
```

### Building input pipelines using tf.data – the TensorFlow Dataset API

If the data already exists in the form of a tensor object, a Python list, or a NumPy array, we can easily create a dataset using the tf.data.Dataset.from_tensor_ slices() function. This function returns an object of class Dataset, which we can use to iterate through the individual elements in the input dataset. As a simple example, consider the following code, which creates a dataset from a list of values:

```python
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
#if we want create batches from dataset (size=3)
ds_batch = ds.batch(3)
```

### Combining two tensors into a joint dataset

Often, we may have the data in two (or possibly more) tensors. For example, we could have a tensor for features and a tensor for labels. In such cases, we need to build a dataset that combines these tensors together, which will allow us to retrieve the elements of these tensors in tuples.

Assume that we have two tensors, t_x and t_y. Tensor t_x holds our feature values, each of size 3, and t_y stores the class labels. For this example, we first create these two tensors as follows:

```python
t_x = tf.random.uniform([4,3], dtype=tf.float32)
t_y = tf.range(4)
#joint dataset
ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))

ds_joint = tf.data.Dataset.from_tensor_slices((ds_x, ds_y))
```

Note that a common source of error could be that the element-wise correspondence between the original features (x) and labels (y) might be lost (for example, if the two datasets are shuffled separately). However, once they are merged into one dataset, it is safe to apply these operations

### Shuffle, batch and repeat

```python
ds = ds_joint.shuffle(buffer_size=len(t_x))

ds = ds_joint.batch(batch_size=3, drop_remainder=False)
batch_x, batch_y = next(iter(ds))

#order of function may be changed
#first batch ds, then repeat (copy) this batches and then shuffled them
ds = ds_joint.batch(2).repeat(3).shuffle(4)

for i, (batch_x, batch_y) in enumerate(ds):
  print(i, batch_x.shape, batch_y.numpy())
```

### Load dataset

```python
#first approach
celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
datasets = celeba_bldr.as_dataset(shuffle_files=False)
#second approach - load combines the three steps in one
mnist, mnist_info = tfds.load('mnist', with_info=True, shuffle_files=False)
```

### Building an NN model in TensorFlow

```python
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

X_train_norm = (X_train - X_train.mean())/X_train.std()
ds = tf.data.Dataset.from_tensor_slices(tf.cast(X_train_norm, dtype=tf.float32), tf.cast(y_train, dtype=tf.float32))

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.w = tf.Variable(0.0, name='weight')
    self.b = tf.Variable(0.0, name='bias')
  def call(self, x):
    return self.w * x + self.b

def loss_fn(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_true - y_pred))

def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as tape:
    current_loss = loss_fn(model(inputs), outputs)
  dw, db = tape.gradient(current_loss, [model.w, model.b])
  model.w.assign_sub(learning_rate * dw)
  model.b.assign_sub(learning_rate * db)

model = MyModel()
model.build(input_shape=(None, 1))
model.summary()
tf.random.set_seed(1)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)
Ws, bs = [], []

for i, batch in enumerate(ds_train):
  if i >= steps_per_epoch * num_epochs:
    # break the infinite loop
    break
  Ws.append(model.w.numpy())
  bs.append(model.b.numpy())
  bx, by = batch
  loss_val = loss_fn(model(bx), by)
  train(model, bx, by, learning_rate=learning_rate)
  if i%log_steps==0:
    print('Epoch {:4d} Step {:2d} Loss {:6.4f}'.format(int(i/steps_per_epoch), i, loss_val))
```

### Tensorflow graph

**Creating a graph in TensorFlow v1.x** 

In the earlier version of the TensorFlow (v1.x) low-level API, this graph had to be explicitly declared. The individual steps for building, compiling, and evaluating such a computation graph in TensorFlow v1.x are as follows: 

1. Instantiate a new, empty computation graph 
2. Add nodes (tensors and operations) to the computation graph 
3. Evaluate (execute) the graph: a. Start a new session b. Initialize the variables in the graph c. Run the computation graph in this session

```python
g = tf.Graph() 
'''
if we do not create a graph, there is default graph to which variables and computations will be added automatically
'''
with g.as_default(): #added nodes to graph
	a = tf.constant(1, name='a')
	b = tf.constant(2, name='b')
	c = tf.constant(3, name='c')
	z = 2*(a-b) + c
```

**Migrating a graph to TensorFlow v2 **

Next, let's look at how this code can be migrated to TensorFlow v2. TensorFlow v2 uses dynamic (as opposed to static) graphs by default (this is also called eager execution in TensorFlow), which allows us to evaluate an operation on the fly. Therefore, we do not have to explicitly create a graph and a session, which makes the development workflow much more convenient:

```python
a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')

z = 2*(a-b) + c
```

### Loading input data into model

**Loading input data into a model: TensorFlow v1.x style**

 Another important improvement from TensorFlow v1.x to v2 is regarding how data can be loaded into our models. In TensorFlow v2, we can directly feed data in the form of Python variables or NumPy arrays. However, when using the TensorFlow v1.x low-level API, we had to create placeholder variables for providing input data to a model.

```python
g = tf.Graph()

with g.as_default():
    a = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_a')
    b = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_b')
    c = tf.compat.v1.placeholder(shape=None, dtype=tf.int32, name='tf_c')
    
    z = 2*(a-b)+c
with tf.compat.v1.Session(graph=g) as sess:
    feed_dict = {a:1, b:2, c:3}
    print(sess.run(z, feed_dict=feed_dict))
```

**Loading input data into a model: TensorFlow v2 style **

In TensorFlow v2, all this can simply be done by defining a regular Python function with a, b, and c as its input arguments, for example:

```python
def compute_z(a, b, c):
	r1 = tf.subtract(a, b)
	r2 = tf.multiply(2, r1)
	z = tf.add(r2, c)
	return z
```

### Improving computational performance with function decoration

```python
'''
@tf.function compile normal python function to a
static TensorFlow graph in order to make computations
more efficient
'''

@tf.function
def compute_z(a, b, c):
  r1 = tf.subtract(a, b)
  r2 = tf.multiply(2, r1)
  z = tf.add(r2, c)
  return z
```

**TensorFlow Variable**

```python
a = tf.Variable(initial_value=3.14, name='var_a')
w = tf.Variable([1, 2, 3], trainable=False)
print(w.assign([3, 1, 4], read_value=True))
w.assign_add([2, -1, 2], read_value=False)
# to initialize weights of NN we can generate a random numbers based on different distrib.
init = tf.keras.initializers.GlorotNormal()
v = tf.Variable(init(shape=(2,3)))

```

When the read_value argument is set to True (which is also the default), these operations will automatically return the new values after updating the current values of the Variable. Setting the read_value to False will suppress the automatic return of the updated value (but the Variable will still be updated in place). 

### Computing the gradients of the loss with respect to trainable variables

Let's work with a simple example where we will compute 𝑧 = 𝑤x+ 𝑏 and define the loss as the squared loss between the target and prediction, 𝐿 = (𝑦 − 𝑧)^2. 

```python
w = tf.Variable(1.0)
b = tf.Variable(0.5)

x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])
with tf.GradientTape() as tape: #persistent=True - to compute gradient more than once
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y-z))

dloss_dw = tape.gradient(loss, w)
print(dloss_dw)
```

### Writing custom Keras layers

In cases where we want to define a new layer that is not already supported by Keras, we can define a new class derived from the tf.keras.layers.Layer class. This is especially useful when designing a new layer or customizing an existing layer.

```python
class NoisyLinear(tf.keras.layers.Layer):
  def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
    self.output_dim = output_dim
    self.noise_stddev = noise_stddev
    super(NoisyLinear, self).__init__(**kwargs)
  
  def build(self, input_shape):
    self.w = self.add_weight(name='weight', 
                             shape=(input_shape[1], self.output_dim),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.output_dim, ),
                             initializer='zeros',
                             trainable=True)
  
  def call(self, inputs, training=False):
    if training:
      batch = tf.shape(inputs)[0]
      dim = tf.shape(inputs)[1]
      noise = tf.random.normal(shape=(batch, dim),
                               mean=0.0,
                               stddev=self.noise_stddev)
      noisy_inputs = tf.add(inputs, noise)
    else:
      noisy_inputs = inputs
    
    z = tf.matmul(noisy_inputs, self.w) + self.b
    return tf.keras.activations.relu(z)

  def get_config(self):
    config = super(NoisyLinear, self).get_config()
    config.update({'output_dim':self.output_dim,
                   'noise_stddev':self.noise_stddev})
    return config

model = tf.keras.Sequential([
    NoisyLinear(4, noise_stddev=0.1),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')])
```

### TensorFlow estimators

Using pre-made Estimators can be summarized in four steps: 

1. Define an input function for data loading 

   ```python
   def train_input_fn(df_train, batch_size=8):
   	df = df_train.copy()
       train_x, train_y = df, df.pop('MPG')
       dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
       # shuffle, repeat, and batch the examples.
       return dataset.shuffle(1000).repeat().batch(batch_size)
   
   def eval_input_fn(df_test, batch_size=8):
       df = df_test.copy()
       test_x, test_y = df, df.pop('MPG')
       dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
       return dataset.batch(batch_size)
   ```

2. Convert the dataset into feature columns 

   ```python
   all_feature_columns = (
       numeric_features +
       bucketized_features +
       categorical_indicator_features)
   ```

3. Instantiate an Estimator (use a pre-made Estimator or create a new one, for example, by converting a Keras model into an Estimator) 

   ```python
   regressor = tf.estimator.DNNRegressor(
       feature_columns=all_feature_columns,
       hidden_units=[32, 10],
       model_dir='models/autompg-dnnregressor/')
   ```

4. Use the Estimator methods train(), evaluate(), and predict()