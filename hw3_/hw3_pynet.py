import numpy as np
from math import *

'''
    Linear

    Implementation of the linear layer (also called fully connected layer),
    which performs linear transformation on input data: y = xW + b.

    This layer has two learnable parameters:
        weight of shape (input_channel, output_channel)
        bias   of shape (output_channel)
    which are specified and initalized in the init_param() function.

    In this assignment, you need to implement both forward and backward
    computation.

    Arguments:
        input_channel  -- integer, number of input channels
        output_channel -- integer, number of output channels
'''
class Linear(object):

    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.init_param()
        self.cache = {}
        ## dictionary to store intermediate value

    def init_param(self):
        self.weight = (np.random.randn(self.input_channel,self.output_channel) * sqrt(2.0/(self.input_channel+self.output_channel))).astype(np.float32)
        self.bias = np.zeros((self.output_channel))

    '''
        Forward computation of linear layer. (3 points)

        Note:  You may want to save some intermediate variables to class
        membership (self.) for reuse in backward computation.

        Arguments:
            input  -- numpy array of shape (N, input_channel)

        Output:
            output -- numpy array of shape (N, output_channel)
    '''
    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        output = input@self.weight+self.bias
        self.cache = (input,self.weight,self.bias)
        return output
    def backward(self, grad_output):
        ## given DL/DZ here
        ## Z =WA+b
        ########################
        # TODO: YOUR CODE HERE #
        ########################A_prev
        A_prev,W,b = self.cache
        m = A_prev.shape[1]
        grad_input = (grad_output@np.transpose(W))
        ## DL/DA = W*DL/DZ
        grad_weight = (np.transpose(A_prev)@(grad_output))
        ## DL/DW = 1/m*DL/DZ*A

        grad_bias = grad_output

        return grad_input, grad_weight, grad_bias

'''
    BatchNorm1d

    Implementation of batch normalization (or BN) layer, which performs
    normalization and rescaling on input data.  Specifically, for input data X
    of shape (N,input_channel), BN layers first normalized the data along batch
    dimension by the mean E(x), variance Var(X) that are computed within batch
    data and both have shape of (input_channel).  Then BN re-scales the
    normalized data with learnable parameters beta and gamma, both having shape
    of (input_channel).
    So the forward formula is written as:

        Y = ((X - mean(X)) /  sqrt(Var(x) + eps)) * gamma + beta

    At the same time, BN layer maintains a running_mean and running_variance
    that are updated (with momentum) during forward iteration and would replace
    batch-wise E(x) and Var(x) for testing. The equations are:

        running_mean = (1 - momentum) * E(x)   +  momentum * running_mean
        running_var =  (1 - momentum) * Var(x) +  momentum * running_var

    During test time, since the batch size could be arbitrary, the statistics
    for a batch may not be a good approximation of the data distribution.
    Thus, we instead use running_mean and running_var to perform normalization.
    The forward formular is modified to:

        Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta

    Overall, BN maintains 4 learnable parameters with shape of (input_channel),
    running_mean, running_var, beta, and gamma.  In this assignment, you need
    to complete the forward and backward computation and handle the cases for
    both training and testing.

    Arguments:
        input_channel -- integer, number of input channel
        momentum      -- float,   the momentum value used for the running_mean and running_var computation
'''
class BatchNorm1d(object):

    def __init__(self, input_channel, momentum = 0.9):
        self.input_channel = input_channel
        self.momentum = momentum
        self.eps = 1e-3
        self.init_param()
        self.cache = {}

    def init_param(self):
        self.r_mean = np.zeros((self.input_channel)).astype(np.float32)
        self.r_var = np.ones((self.input_channel)).astype(np.float32)
        self.beta = np.zeros((self.input_channel)).astype(np.float32)
        self.gamma = (np.random.rand(self.input_channel) * sqrt(2.0/(self.input_channel))).astype(np.float32)

    '''
        Forward computation of batch normalization layer and update of running
        mean and running variance. (3 points)

        You may want to save some intermediate variables to class membership
        (self.) and you should take care of different behaviors during training
        and testing.

        Arguments:
            input -- numpy array (N, input_channel)
            train -- bool, boolean indicator to specify the running mode, True for training and False for testing
    '''
    def forward(self, input, train):
        N = len(input[0])
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        if train == True:
        ## trainning
            E_x = np.mean(input,axis = 0)
            var_x = np.var(input,axis = 0)
            output = self.beta+ self.gamma*((input - E_x)/np.sqrt(var_x+self.eps))
            self.cache = (input,E_x,var_x,output,self.gamma,self.beta)
            return output

    # At the same time, BN layer maintains a running_mean and running_variance
    # that are updated (with momentum) during forward iteration and would replace
    # batch-wise E(x) and Var(x) for testing. The equations are:
    #     running_mean = (1 - momentum) * E(x)   +  momentum * running_mean
    #     running_var =  (1 - momentum) * Var(x) +  momentum * running_var

    # During test time, since the batch size could be arbitrary, the statistics
    # for a batch may not be a good approximation of the data distribution.
    # Thus, we instead use running_mean and running_var to perform normalization.
    # The forward formular is modified to:
    #     Y = ((X - running_mean) /  sqrt(running_var + eps)) * gamma + beta
        else:
        ## testing
            E_x = np.mean(input,axis = 0)
            var_x = np.var(input,axis = 0)
            self.r_mean = (1-self.momentum)*self.r_mean + self.momentum*self.r_mean
            self.r_var = (1-self.momentum)*self.r_var + self.momentum*self.r_var
            output = self.beta+ self.gamma*((input - self.r_mean)/np.sqrt(self.r_var+self.eps))
            self.cache = (input,E_x,var_x,output,self.gamma,self.beta)
            return output

    '''output
        Backward computationg of batch normalization layer. (3 points)
        You need to compute gradient w.r.t input data, gamma, and beta.

        It is recommend to follow the chain rule to first compute the gradient
        w.r.t to intermediate variables, in order to simplify the computation.

        Arguments:
            grad_output -- numpy array of shape (N, input_channel)

        Output:
            grad_input -- numpy array of shape (N, input_channel), gradient w.r.t input
            grad_gamma -- numpy array of shape (input_channel), gradient w.r.t gamma
            grad_beta  -- numpy array of shape (input_channel), gradient w.r.t beta
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N = len(grad_output[0])
        h,mu,var,_,gamma,beta = self.cache
        dy = grad_output
        grad_beta = np.sum(dy,axis = 0)
        grad_gamma = np.sum((h-mu)*(var+self.eps)**(-0.5)*dy,axis = 0)
        temp = (h-mu)*(var+self.eps)**(-0.5)
        dx = dy*gamma
        grad_input = temp/N*(-temp*np.sum(dx*temp,axis = 0))
        return grad_input, grad_gamma, grad_beta
    

'''
    ReLU

    Implementation of ReLU (rectified linear unit) layer.  ReLU is the
    non-linear activation function that sets all negative values to zero.
    The formua is: y = max(x,0).

    This layer has no learnable parameters and you need to implement both
    forward and backward computation.

    Arguments:
        None
'''
class ReLU(object):
    def __init__(self):
        self.cache = {}

    '''
        Forward computation of ReLU. (3 points)

        You may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input  -- numpy array of arbitrary shape

        Output:
            output -- numpy array having the same shape as input.
    '''
    def forward(self, input):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        A = np.maximum(0,input)
        self.cache = input
        return A

    '''
        Backward computation of ReLU. (3 points)

        You can either modify grad_output in-place or create a copy.

        Arguments:
            grad_output -- numpy array having the same shape as input

        Output:
            grad_input  -- numpy array has the same shape as grad_output. gradient w.r.t input
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        grad_input = np.array(grad_output,copy = True)
        A = self.cache
        grad_input[A<=0] = 0
        return grad_input

'''
    CrossEntropyLossWithSoftmax

    Implementation of the combination of softmax function and cross entropy
    loss.  In classification tasks, we usually first apply the softmax function
    to map class-wise prediciton scores into a probability distribution over
    classes.  Then we use cross entropy loss to maximise the likelihood of
    the ground truth class's prediction.  Since softmax includes an exponential
    term and cross entropy includes a log term, we can simplify the formula by
    combining these two functions together, so that log and exp operations
    cancel out.  This way, we also avoid some precision loss due to floating
    point numerical computation.

    If we ignore the index on batch size and assume there is only one grouth
    truth per sample, the formula for softmax and cross entropy loss are:

        Softmax: prob[i] = exp(x[i]) / \sum_{j}exp(x[j])
        Cross_entropy_loss:  - 1 * log(prob[gt_class])

    Combining these two functions togther, we have:

        cross_entropy_with_softmax: -x[gt_class] + log(\sum_{j}exp(x[j]))

    In this assignment, you will implement both forward and backward
    computation.

    Arguments:
        None
'''
class CrossEntropyLossWithSoftmax(object):
    def __init__(self):
        self.cache = {}

    '''
        Forward computation of cross entropy with softmax. (3 points)

        Tou may want to save some intermediate variables to class membership
        (self.)

        Arguments:
            input    -- numpy array of shape (N, C), the prediction for each class, where C is number of classes
            gt_label -- numpy array of shape (N), it is an integer array and the value range from 0 to C-1 which
                        specify the ground truth class for each input
        Output:
            output   -- numpy array of shape (N), containing the cross entropy loss on each input
    '''
    def forward(self, input, gt_label):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        N = len(input[0])
        print(input.shape)
        output = np.log(np.sum(np.exp(input),axis = 1))
        temp = np.zeros((np.shape(output)))
        for i in range(len(gt_label)):
            temp[i] = -input[i][gt_label[i]]         
        output = temp + output
        self.cache = (input,output,gt_label)
        return output

    '''
        Backward computation of cross entropy with softmax. (3 points)

        It is recommended to resue the variable(s) in forward computation
        in order to simplify the formula.

        Arguments:
            grad_output -- numpy array of shape (N)

        Output:
            output   -- numpy array of shape (N, C), the gradient w.r.t input of forward function
    '''
    def backward(self, grad_output):
        ########################
        # TODO: YOUR CODE HERE #
        ########################
        ## dA : post-activation gradient
        
        input,output,gt_label = self.cache
        grad_input = np.exp(input)/(np.sum(np.exp(input),axis = 1))[:,None]
        temp = np.zeros(np.shape(input))
        for i in range(len(gt_label)):
            temp[i][gt_label[i]] = -1  
        grad_input += temp
        return grad_input

'''
    im2col (3 points)

    Consider 4 dimensional input tensor with shape (N, C, H, W), where:
        N is the batch dimension,
        C is the channel dimension, and
        H, W are the spatial dimensions.

    The im2col functions flattens each slidding kernel-sized block
    (C * kernel_h * kernel_w) on each sptial location, so that the output has
    the shape of (N, (C * kernel_h * kernel_w), out_H, out_W) and we can thus
    formuate the convolutional operation as matrix multiplication.

    The formula to compute out_H and out_W is the same as to compute the output
    spatial size of a convolutional layer.

    Arguments:
        input_data  -- numpy array of shape (N, C, H, W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- numpy array of shape (N, (C * kernel_h * kernel_w), out_H, out_W)
'''
def im2col(input_data, kernel_h, kernel_w, stride, padding):
    ########################
    # TODO: YOUR CODE HERE #
    ########################
    
    N, C, H, W = np.shape(input_data)
    out_H = int(((H-kernel_h + 2 * padding) / stride) + 1)
    out_W = int(((W-kernel_w+ 2 * padding) / stride) + 1)
    output_data = np.zeros((N,C * kernel_h * kernel_w,out_H,out_W))
    ## create output frame
    X_pad = np.pad(input_data, (padding, padding), 'constant', constant_values=(0, 0))[padding:N+padding, padding:C+padding, :, :]
    ## padding first
###  loop over N
    for i in range(N):
        ## loop over channel
            for h in range(out_H):
                for w in range(out_W):
                    top = h*stride
                    down = kernel_h+top
                    left = w*stride
                    right = kernel_w+left
                    slice_temp = X_pad[i,:,top:down,left:right]
                    ## slice shape c*kernel*kernel
                    output_data[i,:,h,w] = slice_temp.reshape((C*kernel_h*kernel_w))

    return output_data

'''
    col2im (3 points)

    Consider a 4 dimensional input tensor with shape:
        (N, (C * kernel_h * kernel_w), out_H, out_W)
    where:
        N is the batch dimension,
        C is the channel dimension,
        out_H, out_W are the spatial dimensions, and
        kernel_h and kernel_w are the specified kernel spatial dimension.

    The col2im function calculates each combined value in the resulting array
    by summing all values from the corresponding sliding kernel-sized block.
    With the same parameters, the output should have the same shape as
    input_data of im2col.  This function serves as an inverse subroutine of
    im2col, so that we can formuate the backward computation in convolutional
    layers as matrix multiplication.

    Arguments:
        input_data  -- numpy array of shape (N, (C * kernel_H * kernel_W), out_H, out_W)
        kernel_h    -- integer, height of the sliding blocks
        kernel_w    -- integer, width of the sliding blocks
        stride      -- integer, stride of the sliding block in the spatial dimension
        padding     -- integer, zero padding on both size of inputs

    Returns:
        output_data -- output_array with shape (N, C, H, W)
'''
def col2im(input_data, kernel_h, kernel_w, stride, padding, C, H, W):
    ########################
    N, out, out_H, out_W = input_data.shape
    output_data = np.zeros((N, C, H, W))
    for i in range(N):
        for h in range(out_H):
            for w in range(out_W):
                slice1 =  input_data[i, :, h, w]
                slice_temp =np.reshape(slice1,(C, kernel_h, kernel_w))
                top = max(0, h*stride-padding)
                if  (h*stride-padding) < 0:
                    down = h*stride-padding+kernel_h
                else:
                    down = top+kernel_h            
                down = min(H, down)
                down1 = min(0,H-down)+kernel_h
                left = max(0, w*stride-padding)
                left1 = max(0,padding-w*stride)
                top1 = max(0,padding-h*stride)
                if (w*stride-padding) < 0:
                    right = kernel_w+(w*stride-padding)
                else:
                    right = kernel_w+left
                right = min(W, right)
                right1 = min(0,W-right)+kernel_w

                output_data[i, :, top:down, left:right] += slice_temp[:, top1:down1, left1:right1]
    return output_data

'''
    Conv2d

    Implementation of convolutional layer.  This layer performs convolution
    between each sliding kernel-sized block and convolutional kernel.  Unlike
    the convolution you implemented in HW1, where you needed flip the kernel,
    here the convolution operation can be simplified as cross-correlation (no
    need to flip the kernel).

    This layer has 2 learnable parameters, weight (convolutional kernel) and
    bias, which are specified and initalized in the init_param() function.
    You need to complete both forward and backward functions of the class.
    For backward, you need to compute the gradient w.r.t input, weight, and
    bias.  The input arguments: kernel_size, padding, and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You need to use im2col, col2im inside forward and backward respectively,
    which formulates the sliding window computation in a convolutional layer as
    matrix multiplication.

    Arguments:
        input_channel  -- integer, number of input channel which should be the same as channel numbers of filter or input array
        output_channel -- integer, number of output channel produced by convolution or the number of filters
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class Conv2d(object):
    def __init__(self, input_channel, output_channel, kernel_size, padding = 0, stride = 1):
        self.output_channel = output_channel
        self.input_channel = input_channel
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.init_param()
        self.cache = {}

    def init_param(self):
        self.weight = (np.random.randn(self.output_channel, self.input_channel, self.kernel_h, self.kernel_w) * sqrt(2.0/(self.input_channel + self.output_channel))).astype(np.float32)
        self.bias = np.zeros(self.output_channel).astype(np.float32)

    '''
        Forward computation of convolutional layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, output_chanel, out_H, out_W)
    '''
    def forward(self, input):
        N, C, H, W = np.shape(input)
        out_H = int((H-self.kernel_h + 2 * self.padding) / self.stride + 1)
        out_W = int((W-self.kernel_w+ 2 * self.padding) / self.stride + 1)
        slices1 = im2col(input, self.kernel_h, self.kernel_w, self.stride, self.padding)
        output = np.zeros((N,self.output_channel,out_H,out_W))
        
        for i in range(N):
            ## loop over channel
            for c in range(self.output_channel):
                for h in range(out_H):
                    for w in range(out_W):
                        slice_temp = slices1[i,:,h,w]
                        W_temp = self.weight[c,:,:,:]
                        b_temp = self.bias[c]
                        slice_temp = np.reshape(slice_temp,W_temp.shape)
                        ## reshape to the same shape as W
                        ## start convolution
                        temp = slice_temp*W_temp
                        temp1 = np.sum(temp)
                        temp2 = float(temp1+b_temp)
                        output[i,c,h,w] = temp2
        self.cache = (N,C,H,W,out_H,out_W,input,self.weight,slices1)
        return output

    '''
        Backward computation of convolutional layer. (3 points)

        You need col2im and saved variables from forward() in this function.

        Arguments:
            grad_output -- numpy array of shape (N, output_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
            grad_weight -- numpy array of shape(output_channel, input_channel, kernel_h, kernel_w), gradient w.r.t weight
            grad_bias   -- numpy array of shape(output_channel), gradient w.r.t bias
    '''

    def backward(self, grad_output):
        N, C, H, W,out_H,out_W,A_prev,weight,A_prev = self.cache
        ## A_prev shape (N, input_channel, H, W)
        grad_input = np.zeros((N,C * self.kernel_h * self.kernel_w,out_H,out_W))
        grad_weight = np.zeros((self.output_channel,self.input_channel,self.kernel_h,self.kernel_w))
        grad_bias = np.zeros((self.output_channel))
       ## grad_input = np.pad(grad_input, (self.padding, self.padding), 'constant', constant_values=(0, 0))[self.padding:N+self.padding, self.padding:C+self.padding, :, :]
        ## A_prev = np.pad(A_prev, (self.padding, self.padding), 'constant', constant_values=(0, 0))[self.padding:N+self.padding, self.padding:C+self.padding, :, :]
        for i in range(N):
            ## loop over channel
            for c in range(self.output_channel):
                for h in range(out_H):
                    for w in range(out_W):
                        top = h
                        bottom = h+self.kernel_h
                        left = w
                        right = w+self.kernel_w
                        slice_temp = A_prev[i,:,h,w].reshape(self.input_channel,self.kernel_h,self.kernel_w)
                        grad_input[i,:,h,w] += (weight[c,:,:,:]*grad_output[i,c,h,w]).flatten()
                        grad_weight[c,:,:,:] += slice_temp*grad_output[i,c,h,w]
                        grad_bias[c] += grad_output[i,c,h,w]
        grad_input = col2im(grad_input, self.kernel_h, self.kernel_w, self.stride, self.padding, C, H, W)
        return grad_input, grad_weight, grad_bias

'''
    MaxPool2d

    Implementation of max pooling layer.  For each sliding kernel-sized block,
    maxpool2d computes the spatial maximum along each channels.  This layer has
    no learnable parameters.

    You need to complete both forward and backward functions of the layer.
    For backward, you need to compute the gradient w.r.t input.  Similar as
    conv2d, the input argument, kernel_size, padding and stride jointly
    determine the output shape by the following formula:

        out_shape = (input_shape - kernel_size + 2 * padding) / stride + 1

    You may use im2col, col2im inside forward and backward, respectively.

    Arguments:
        kernel_size    -- integer or tuple, spatial size of convolution kernel. If it's tuple, it specifies the height and
                          width of kernel size.
        padding        -- zero padding added on both sides of input array
        stride         -- integer, stride of convolution.
'''
class MaxPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride

    '''
        Forward computation of max pooling layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, input_channel, out_H, out_W)
    '''
class MaxPool2d(object):
    def __init__(self, kernel_size, padding = 0, stride = 1):
        if isinstance(kernel_size, tuple):
            self.kernel_h, self.kernel_w = kernel_size
        else:
            self.kernel_w = self.kernel_h = kernel_size
        self.padding = padding
        self.stride = stride
        self.cache = {}

    '''
        Forward computation of max pooling layer. (3 points)

        You should use im2col in this function.  You may want to save some
        intermediate variables to class membership (self.)

        Arguments:
            input   -- numpy array of shape (N, input_channel, H, W)

        Ouput:
            output  -- numpy array of shape (N, input_channel, out_H, out_W)
    '''
    def forward(self, input):
        N, C, H, W = np.shape(input)
        out_H = int((H-self.kernel_h + 2 * self.padding) / self.stride) + 1
        out_W = int((W-self.kernel_w+ 2 * self.padding) / self.stride) + 1
        slices1 = im2col(input, self.kernel_h, self.kernel_w, self.stride, self.padding)
        output = np.zeros((N,C,out_H,out_W))

        for i in range(N):
            ## loop over channel
            for c in range(C):
                for h in range(out_H):
                    for w in range(out_W):
                        slice_temp = slices1[i,:,h,w]
                        slice_temp = slice_temp.reshape(C,self.kernel_h,self.kernel_w)
                        ## reshape to the same shape as W
                        temp2 = np.max(slice_temp,axis = 1)
                        temp2 = np.max(temp2,axis = 1)
                        output[i,:,h,w] = temp2
        self.cache = (slices1,N,C,H,W,out_H,out_W)
        return output

    '''
        Backward computation of max pooling layer. (3 points)

        You should use col2im and saved variable(s) from forward().

        Arguments:
            grad_output -- numpy array of shape (N, input_channel, out_H, out_W)

        Ouput:
            grad_input  -- numpy array of shape(N, input_channel, H, W), gradient w.r.t input
    '''
    def backward(self, grad_output):
        (input,N,C,H,W,out_H,out_W) = self.cache
        grad_input = np.zeros_like(input)
        for i in range(N):
                for h in range(out_H):
                    for w in range(out_W):
                        slice_temp = input[i,:,h,w].reshape((C,self.kernel_w,self.kernel_h))
                        mask = np.zeros((C,self.kernel_w,self.kernel_h))
                        temp = (np.max(np.max(slice_temp,axis = 1),axis = 1))
                        for m in range(C):
                            mask[m,:,:] = (temp[i] == slice_temp[i,:,:])
                        mask = np.array(mask,dtype = int)
                        ##mask = (slice_temp == (np.max(np.max(slice_temp,axis = 1),axis = 1)))
                        temp = np.zeros(np.shape(mask))
                        for dim in range(np.shape(mask)[0]):
                            temp[dim] = (mask[dim]*grad_output[i,:,h,w][dim])
                        grad_input[i,:,h,w] = temp.flatten()
        grad_input = col2im(grad_input, self.kernel_h, self.kernel_w, self.stride, self.padding, C, H, W)
        return grad_input

