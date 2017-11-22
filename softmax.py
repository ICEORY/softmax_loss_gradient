"""
softmax
"""
import torch
from torch.autograd import Variable
from torch.autograd import Function

def test_softmax_forward():
    x = torch.randn(1, 3)
    x_var = Variable(x)
    y = SoftMax()(x_var)
    y_hat = F.softmax(x_var)
    dist = (y_hat - y).data.abs().sum()

    print "|===> test softmax forward"
    print y
    print y_hat
    print "dist: ", dist


def test_softmax_backward():
    x = torch.ones(1, 3)
    x_var = Variable(x, requires_grad=True)
    y = SoftMax()(x_var)
    x_copy = x.clone()
    x_var_copy = Variable(x_copy, requires_grad=True)
    y_hat = F.softmax(x_var_copy)
    y.sum().backward()
    y_hat.sum().backward()
    dist = (x_var.grad - x_var_copy.grad).data.abs().sum()

    print "|===> test softmax backward"
    print "y: ", y
    # print "y.grad", y.grad
    print "x_var.grad: ", x_var.grad
    print "y_hat",  y_hat
    # print "y_hat.grad: ", y_hat.grad
    print "x_var_copy.grad: ", x_var_copy.grad
    print "dist: ", dist


class SoftMax(Function):
    """
    soft_max with custom forward and backward propagation

    """

    def forward(self, x):
        r"""
        forward propagation
        function:
        y(x_i) =\frac{exp(x_i)}{\sum_{k=1}^K exp(x_k)}~~for j=1,\dots, K
        """
        assert x.dim() == 2, "dimension of input should be 2"
        exp_x = torch.exp(x)
        output = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
        self.output = output
        return output

    def backward(self, grad_output):
        r"""
        backward propagation
        gradient:
        \frac{\partial y_i}{x_i} = y_i (1-y_i) if i=j
        \frac{\partial y_i}{x_j} = -y_i*y_j if i \ne j
        suppose y_i = softmax(x_i)
        """
        pass
        # return grad_input



