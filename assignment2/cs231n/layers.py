from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    D = w.shape[0]
    reshaped_x = np.reshape(x, (N, D))
    out = reshaped_x.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    D = w.shape[0]
    reshaped_x = np.reshape(x, (N, D))

    dx = dout.dot(w.T).reshape(x.shape)
    dw = reshaped_x.T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = dout * (1 * (cache > 0)), cache
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    cache = {}
    if mode == 'train':
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        xhat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma*xhat + beta

        cache['xhat'] = xhat
        cache['var'] = sample_var + eps
        cache['gamma'] = gamma

        running_mean = momentum*running_mean + (1-momentum)*sample_mean
        running_var = momentum*running_var + (1-momentum)*sample_var
    elif mode == 'test':
        xhat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma*xhat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    N = dout.shape[0]

    istd = 1/np.sqrt(cache['var'])
    x_minus_mean = cache['xhat'] / istd

    dxhat = dout * cache['gamma']
    dvar = np.sum(dxhat * (-.5)*x_minus_mean * istd**3, axis=0)
    dmean = np.sum(dxhat * -istd, axis=0) + \
        dvar * (-2/N)*np.sum(x_minus_mean, axis=0)
    dx = (dxhat * istd) + \
         (dvar * (2/N)*x_minus_mean) + \
         (dmean / N)

    dgamma = np.sum(dout * cache['xhat'], axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    N = dout.shape[0]
    dgamma = np.sum(dout * cache['xhat'], axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = (cache['gamma'] * (1 / np.sqrt(cache['var'])) / N) * (N * dout - cache['xhat'] * dgamma - dbeta)

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    cache = {}
    eps = ln_param.get('eps', 1e-5)
    sample_mean = np.mean(x.T, axis=0)
    sample_var = np.var(x.T, axis=0)
    xhat = (x - sample_mean[:, np.newaxis]) / np.sqrt(sample_var[:, np.newaxis] + eps)
    out = gamma * xhat + beta

    cache['xhat'] = xhat
    cache['var'] = sample_var + eps
    cache['gamma'] = gamma

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dgamma = np.sum(dout * cache['xhat'], axis=0)
    dbeta = np.sum(dout, axis=0)
    dxhat = dout * cache['gamma']

    D = dout.shape[1]
    dx = (1 / np.sqrt(cache['var'])) / D * (
            D*dxhat.T - cache['xhat'].T*np.sum(dxhat.T * cache['xhat'].T, axis=0) - np.sum(dxhat.T, axis=0))

    return dx.T, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = mask * dout
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    H_prime = int(1 + (H + 2*pad - HH) / stride)
    W_prime = int(1 + (W + 2*pad - WW) / stride)
    out = np.zeros((N, F, H_prime, W_prime))

    for i in range(H_prime):
        for j in range(W_prime):
            for f in range(F):
                conv_filter = w[f, :, :, :]
                h_index = i*stride
                w_index = j*stride
                data = padded_x[:, :, h_index:(h_index+HH), w_index:(w_index+WW)]
                out[:, f, i, j] = np.sum(np.multiply(conv_filter, data), axis=(1, 2, 3)) + b[f]


    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)

    db = np.zeros(b.shape)
    for i in range(F):
        db[i] = np.sum(dout[:, i, :, :])

    dw = np.zeros(w.shape)
    for f in range(F):
        for c in range(C):
            for i in range(HH):
                for j in range(WW):
                    dw[f, c, i, j] = \
                        np.sum(dout[:, f, :, :] *
                               padded_x[:, c, i:i + H_prime*stride:stride, j:j + W_prime*stride:stride])

    dx = np.zeros(x.shape)
    for n in range(N):
        dx_pad = np.pad(dx[n, :, :, :], ((0, 0), (pad, pad), (pad, pad)), 'constant')
        for f in range(F):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):
                    h1 = h_prime * stride
                    h2 = h_prime * stride + HH
                    w1 = w_prime * stride
                    w2 = w_prime * stride + WW
                    dx_pad[:, h1:h2, w1:w2] += w[f, :, :, :] * dout[n, f, h_prime, w_prime]
        dx[n, :, :, :] = dx_pad[:, 1:-1, 1:-1]


    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    stride = pool_param['stride']
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    H_prime = int(1 + (H - HH) / stride)
    W_prime = int(1 + (W - WW) / stride)
    out = np.zeros((N, C, H_prime, W_prime))

    for i in range(H_prime):
        for j in range(W_prime):
            h_index = i * stride
            w_index = j * stride
            data = x[:, :, h_index:(h_index + HH), w_index:(w_index + WW)]
            out[:, :, i, j] = np.max(data, axis=(2, 3))

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    stride = pool_param['stride']
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    H_prime = int(1 + (H - HH) / stride)
    W_prime = int(1 + (W - WW) / stride)
    dx = np.zeros(x.shape)

    for n in range(N):
        for c in range(C):
            for i in range(H_prime):
                for j in range(W_prime):
                    h_index = i * stride
                    w_index = j * stride
                    data = x[n, c, h_index:(h_index + HH), w_index:(w_index + WW)].reshape(-1)
                    max_ind = np.argmax(data)
                    grad = np.zeros(data.shape)
                    grad[max_ind] = dout[n, c, i, j]
                    grad.reshape((HH, WW))
                    dx[n, c, h_index:(h_index + HH), w_index:(w_index + WW)] = grad.reshape((HH, WW))

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x_reshaped = np.reshape(x.transpose((0, 2, 3, 1)), (N*H*W, C))
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = np.reshape(out, (N, H, W, C)).transpose((0, 3, 1, 2))

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dout_reshaped = np.reshape(dout.transpose((0, 2, 3, 1)), (N * H * W, C))
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = np.reshape(dx, (N, H, W, C)).transpose((0, 3, 1, 2))

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """

    eps = gn_param.get('eps', 1e-5)

    N, C, H, W = x.shape
    x_reshaped = x.reshape(N, G, C // G, H, W)
    sample_mean = x_reshaped.mean(axis=(2, 3, 4), keepdims=True)
    sample_var = x_reshaped.var(axis=(2, 3, 4), keepdims=True) + eps

    xhat = (x_reshaped - sample_mean) / np.sqrt(sample_var)
    xhat = xhat.reshape(N, C, H, W)
    out = xhat * gamma + beta

    cache = {'xhat': xhat,
             'gamma': gamma,
             'sample_var': sample_var,
             'G': G}

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    G = cache['G']
    xhat = cache['xhat']

    dgamma = np.sum(dout * xhat, axis=(0, 2, 3)).reshape((1, C, 1, 1))
    dbeta = np.sum(dout, axis=(0, 2, 3)).reshape((1, C, 1, 1))

    dxhat = (dout * cache['gamma']).reshape((N*G, C//G * H*W))
    xhat = xhat.reshape((N*G, C//G * H*W))

    inverse_var = 1/np.sqrt(cache['sample_var'].reshape(N*G))
    D = dxhat.shape[1]

    dx = inverse_var/D * (D*dxhat.T - xhat.T*np.sum(dxhat.T * xhat.T, axis=0) - np.sum(dxhat.T, axis=0))
    dx = np.reshape(dx.T, (N, C, H, W))

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
