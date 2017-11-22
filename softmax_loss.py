"""
softmax with loss
"""
import torch
import time
from torch.autograd import Variable
from torch.autograd import Function

def test_softmax_loss_backward():
    """
    analyse the difference between autograd and manual grad
    """
    x_size = 3200
    # generate random testing data
    x = torch.randn(x_size, x_size).double()
    x_var = Variable(x, requires_grad=True)

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
    t0 = time.time()
    y.backward() # manual gradient
    t1 = time.time()
    y_hat.backward() # auto gradient
    t2 = time.time()

    dist = (y_hat - y).data.abs().sum()
    print "====================================================="
    print "|===> testing softmax loss forward"
    print "distance between y_hat and y: ", dist

    dist = (x_var.grad - x_var_copy.grad).data.abs().sum()
    print "|===> testing softmax loss backward"
    print "y: ", y
    print "y_hat: ", y_hat
    print "x_grad: ", x_var.grad
    print "x_copy.grad: ", x_var_copy.grad
    print "distance between x.grad and x_copy.grad: ", dist

    print "|===> comparing time-costing"
    print "time of manual gradient: ", t1-t0
    print "time of auto gradient: ", t2-t1
    # different dist=1.38e-7 with float precision
    # different dist=3.34e-16 with double precision


def SoftmaxLossFunc(x, target):
    exp_x = torch.exp(x)
    y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
    t = torch.zeros(y.size()).type(y.data.type())
    for n in range(t.size(0)):
        t[n][target.data[n]] = 1

    t = Variable(t)
    output = (-t * torch.log(y)).sum() / y.size(0)
    return output

class SoftmaxLoss(Function):
    r"""
    softmax with cross entropy
    log_softmax:
    y = log(\frac{e^x}{\sum e^{x_k}})
    negative likelyhood:
    z = - \sum t_i y_i, where t is one hot target
    """

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
        self.y = y
        self.t = t
        return output

    def backward(self, grad_output):
        """
        backward propagation
        """
        grad_input = grad_output * (self.y - self.t) / self.y.size(0)
        return grad_input, None
