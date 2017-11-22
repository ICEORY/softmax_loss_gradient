# Read Me
---

## Checking the difference between auto_grad and hand-craft gradient

1. given random input, compute gradient of softmax and cross entropy
2. run experiments on cifar and compute difference of gradient with increasing of depth

## Reproduction of Softmax Loss with Cross Entropy
### softmax function
the softmax function is defined by

$$
y_i = \frac{e^{x_i}}{\sum e^{x_k}}, for~i=1,..., C
$$
where $x$ is the input with $C$ channels, $y$ is the respected output.

the gradient of softmax $\frac{\partial y_i}{\partial x_j}$ is computed by:

$$
{\rm if}~i=j,~\frac{\partial y_i}{\partial x_j}=\frac{\partial y_i}{\partial x_i} = \frac{e^{x_j}\cdot \sum e^{x_k}-e^{x_i}\cdot e^{x_i}}{(\sum e^{x_k})^2} = \frac{e^{x_i}}{\sum e^{x_k}}\frac{\sum e^{x_k}-e^{x_i}}{\sum e^{x_k}} = y_i \cdot (1-y_i)
$$

$$
{\rm if}~i\ne j,~\frac{\partial y_i}{\partial x_j}=\frac{\partial \frac{e^{x_j}}{\sum e^k}}{\partial x_j} = \frac{0\cdot \sum e^{x_k}-e^{x_i}\cdot e^{x_j}}{(\sum e^{x_k})^2} = -\frac{e^{x_i}}{\sum e^{x_k}} \frac{e^{x_j}}{\sum e^{x_k}} = -y_i \cdot y_j
$$

### Cross entropy loss
the cross entropy loss function is 

$$
\mathcal L = -\sum_{c=1}^C t_c\cdot log(y_c),
$$
where $t$ is the one-hot label.

For a batch of samples, the cross-entropy loss can be re-written to

$$
\mathcal L = -\sum_{n=1}^N \sum_{c=1}^C t_{nc} \cdot log(y_{nc})
$$

the gradient of cross entropy loss is computed by

$$
\frac{\partial \mathcal L}{\partial x_i} = -\sum_{c=1}^C \frac{\partial t_c log (y_c)}{\partial x_i} = -\sum_{c=1}^C t_c \frac{\partial log(y_c)}{\partial x_i} = -\sum_{c=1}^C t_c\frac{1}{y_c}\frac{\partial y_c}{\partial x_i} = -\frac{t_i}{y_i}\frac{\partial {y_i}}{\partial x_i}-\sum_{i\ne j}^C \frac{t_j}{y_j}\frac{\partial y_j}{\partial x_i} = -\frac{t_i}{y_i}y_i(1-y_i) - \sum_{i\ne j}^C \frac{\partial t_j}{y_j}(-y_i y_j) = -t_i+t_i y_i + \sum_{i\ne j}^C t_j y_i = -t_i +y_i\sum_i^C t_i = y_i-t_i
$$

### Coding in PyTorch

#### Using basic function of PyTorch

##### forward propagation of SoftMaxLoss

```python
def forward(self, x, target):
        """
        forward propagation
        """
        assert x.dim() == 2, "dimension of input should be 2"
        exp_x = torch.exp(x)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        
        # parameter "target" is a LongTensor and denotes the labels of classes, here we need to convert it into one hot vectors 
        t = torch.zeros(y.size()).type(y.type())
        for n in range(t.size(0)):
            t[n][target[n]] = 1

        output = (-t * torch.log(y)).sum() / y.size(0)
        # output should be a tensor, but the output of sum() is float
        output = torch.Tensor([output]).type(y.type())
        self.y = y # save for backward
        self.t = t # save for backward
        return output
```
to use the auto-grad scheme of PyTorch, we also define a function to execute the same operation of forward propagation of softmax loss
```python
def SoftmaxLossFunc(x, target):
    exp_x = torch.exp(x)
    y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
    t = torch.zeros(y.size()).type(y.data.type())
    for n in range(t.size(0)):
        t[n][target.data[n]] = 1

    t = Variable(t)
    output = (-t * torch.log(y)).sum() / y.size(0)
    return output
```
##### backward propagation:
```python
def backward(self, grad_output):
        """
        backward propagation
        """
        grad_input = grad_output * (self.y - self.t) / self.y.size(0)
        return grad_input, None
```

#### testing code:
```python
def test_softmax_loss_backward():
    """
    analyse the difference between autograd and manual grad
    """
    # generate random testing data
    x_size = 3200
    x = torch.randn(x_size, x_size) # .cuda() # use .cuda for GPU mode
    x_var = Variable(x, requires_grad=True) # convert tensor into Variable

    # testing labels
    target = torch.LongTensor(range(x_size))
    target_var = Variable(target)

    # compute outputs of softmax loss
    y = SoftmaxLoss()(x_var, target_var)

    # clone testing data
    x_copy = x.clone()
    x_var_copy = Variable(x_copy, requires_grad=True)

    # compute output of softmax loss
    y_hat = SoftmaxLossFunc(x_var_copy, target_var)

    # compute gradient of input data with two different method
    y.backward() # manual gradient
    y_hat.backward() # auto gradient
    
    # compute difference of gradients
    grad_dist = (x_var.grad - x_var_copy.grad).data.abs().sum()
```

outputs:

the distance between our implementation and PyTorch auto-gradient is about e-7 under 32 bits floating point precision, and our backward operation is slightly faster than the baseline

```
=====================================================
|===> testing softmax loss forward
distance between y_hat and y:  0.0
|===> testing softmax loss backward
y:  Variable containing:
 8.5553
[torch.FloatTensor of size 1]

y_hat:  Variable containing:
 8.5553
[torch.FloatTensor of size 1]

x_grad:  Variable containing:
-3.1247e-04  1.3911e-07  4.8041e-07  ...   3.0512e-08  1.7696e-08  1.0826e-07
 7.6744e-07 -3.1246e-04  1.2172e-07  ...   1.2465e-07  6.0764e-08  5.0740e-08
 8.7925e-08  1.7995e-08 -3.1242e-04  ...   1.1499e-07  6.7635e-08  5.2739e-08
                ...                   ⋱                   ...                
 1.0118e-08  1.7118e-07  1.7081e-07  ...  -3.1244e-04  3.1381e-07  2.1709e-08
 2.2232e-07  2.4775e-07  1.0417e-07  ...   4.6105e-08 -3.1172e-04  2.1110e-08
 1.6006e-07  4.8581e-08  3.2675e-08  ...   2.3572e-07  5.3878e-08 -3.1247e-04
[torch.FloatTensor of size 3200x3200]

x_copy.grad:  Variable containing:
-3.1247e-04  1.3911e-07  4.8041e-07  ...   3.0512e-08  1.7696e-08  1.0826e-07
 7.6744e-07 -3.1246e-04  1.2172e-07  ...   1.2465e-07  6.0764e-08  5.0740e-08
 8.7925e-08  1.7995e-08 -3.1242e-04  ...   1.1499e-07  6.7635e-08  5.2739e-08
                ...                   ⋱                   ...                
 1.0118e-08  1.7118e-07  1.7081e-07  ...  -3.1244e-04  3.1381e-07  2.1709e-08
 2.2232e-07  2.4775e-07  1.0417e-07  ...   4.6105e-08 -3.1172e-04  2.1110e-08
 1.6006e-07  4.8581e-08  3.2675e-08  ...   2.3572e-07  5.3878e-08 -3.1247e-04
[torch.FloatTensor of size 3200x3200]

distance between x.grad and x_copy.grad:  1.11203504294e-07
|===> comparing time-costing
time of manual gradient:  1.13225889206
time of auto gradient:  1.40407109261

```

with 64 bits double precision, the difference of gradient is reduced into e-16. Notice that the outputs of two sofmaxloss function have a gap of e-7. Again, our method is slightly faster.

```
=====================================================
|===> testing softmax loss forward
distance between y_hat and y:  2.31496107617e-07
|===> testing softmax loss backward
y:  Variable containing:
 8.5468
[torch.DoubleTensor of size 1]

y_hat:  Variable containing:
 8.5468
[torch.DoubleTensor of size 1]

x_grad:  Variable containing:
-3.1246e-04  4.7302e-08  2.6106e-08  ...   2.1885e-08  1.5024e-08  6.0311e-09
 4.1688e-08 -3.1245e-04  1.1503e-07  ...   1.8215e-07  3.1857e-08  1.1914e-07
 9.2476e-08  7.1073e-08 -3.1248e-04  ...   2.7795e-08  2.5479e-07  4.8765e-08
                ...                   ⋱                   ...                
 5.0167e-08  1.2661e-07  8.0579e-08  ...  -3.1239e-04  2.0139e-08  1.3870e-08
 2.8047e-07  3.2061e-07  1.8310e-08  ...   1.5054e-08 -3.1248e-04  8.4565e-08
 5.4617e-08  4.3503e-08  5.2926e-08  ...   1.2573e-07  3.3953e-08 -3.1236e-04
[torch.DoubleTensor of size 3200x3200]

x_copy.grad:  Variable containing:
-3.1246e-04  4.7302e-08  2.6106e-08  ...   2.1885e-08  1.5024e-08  6.0311e-09
 4.1688e-08 -3.1245e-04  1.1503e-07  ...   1.8215e-07  3.1857e-08  1.1914e-07
 9.2476e-08  7.1073e-08 -3.1248e-04  ...   2.7795e-08  2.5479e-07  4.8765e-08
                ...                   ⋱                   ...                
 5.0167e-08  1.2661e-07  8.0579e-08  ...  -3.1239e-04  2.0139e-08  1.3870e-08
 2.8047e-07  3.2061e-07  1.8310e-08  ...   1.5054e-08 -3.1248e-04  8.4565e-08
 5.4617e-08  4.3503e-08  5.2926e-08  ...   1.2573e-07  3.3953e-08 -3.1236e-04
[torch.DoubleTensor of size 3200x3200]

distance between x.grad and x_copy.grad:  1.99762357071e-16
|===> comparing time-costing
time of manual gradient:  1.170181036
time of auto gradient:  2.39760398865

```

### Reference:

[1] http://shuokay.com/2016/07/20/softmax-loss/

[2] https://en.wikipedia.org/wiki/Cross_entropy