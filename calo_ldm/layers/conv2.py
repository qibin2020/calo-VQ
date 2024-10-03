import torch
from torch import nn
import torch.nn.functional as F

# just some error checking for the layer arguments.
def _check_arg_pair(args, name="kernel size", metavar=None):
    if not metavar:
        metavar = name[0]

    try:
        args = tuple((x for x in args))
    except TypeError:
        # if it's not iterable, just duplicate it
        args = (args, args)

    # make sure args are integers
    if not all((isinstance(x, int) for x in args)):
        raise TypeError(f"{name} must be an integer: {args}")

    if len(args) != 2:
        raise ValueError(f"wrong number of {name}s provided: {args}\nPlease specify either a scalar `{metavar}` or tuple `({metavar}_z, {metavar}_phi)`.")

    if any((x < 1 for x in args)): raise ValueError(f"{name}s must be positive: {args}")

    return args


class CylinderConv2(nn.Module):
    '''
    Applies cylindrical convolution to the incoming data.
    
    Args:
      in_features : int
        number of input channels (radial dimension for input layer)
      out_features : int
        number of output channels
      k : int or (int, int), optional
        Scalar k or tuple (k_z, k_phi) for kernel size in z and phi directions.
        If k is a scalar, set (k_z, k_phi) = (k, k).
      strides : int or (int, int), optional
        Scalar s or tuple (s_z, s_phi) for convolution striding in z and phi directions.
        If s is a scalar, set (s_z, s_phi) = (s, s).
        In the radial direction, _s_phi_ must evenly divide the original size.
      pad_z: bool, optional
        If True, pad the Z dimension on the input, such that the output size is not reduced when stride_z=1,
        a.k.a. padding="same".
      bias : bool, optional
        Whether or not to include a trainable bias offset.
    
    Shape:
      Input: (*, C_in, Z_in, Phi_in) where * is any number of dimensions including none.
      Output: (*, C_out, Z_out, Phi_out) where for the simplest case of strides=1 and pad_z=False,
        Z_out = Z_in - (k_z-1)
        Phi_out = Phi_in
    '''
    def __init__(self, in_features, out_features, k=(5,3), stride=(1,1), pad_z=False, bias=True):
        super().__init__()
        
        k = _check_arg_pair(k, "kernel size")
        stride = _check_arg_pair(stride, "stride")
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.stride = stride
        self.pad_z_n=0 if not pad_z else (k[0]-1)//2
        self.pad_a_n=0 if k[1] <= 1 else (k[1]-1)//2

        wshape = (out_features, in_features, k[0], k[1])
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(wshape)))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        # kz, kp = self.k
        if self.pad_a_n > 0:
            # warning! If you try to handle the kp=1 case here, the slices don't work
            # as expected. But kp=1 corresponds to no padding, so just skip it.
            
            a = x[..., :self.pad_a_n ] # chunk from the left side of the phi direction
            b = x[..., -self.pad_a_n:]  # chunk from the right side of the phi direction

            x = torch.concat([b, x, a], dim=-1) # cyclic padding in phi dimension

        x = F.conv2d(x, self.weights, bias=self.bias, stride=self.stride, padding=(self.pad_z_n,0))
        
        return x
    
    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, k={self.k}, stride={self.stride}, pad_z={self.pad_z_n}"
    
class CylinderConvTranspose2(nn.Module):
    '''
    Applies trasposed cylindrical convolution to the incoming data.
    
    Args:
      in_features : int
        number of input channels
      out_features : int
        number of output channels (radial dimension for final layer)
      k : int or (int, int), optional
        Scalar k or tuple (k_z, k_phi) for kernel size in z and phi directions.
        If k is a scalar, set (k_z, k_phi) = (k, k).
      strides : int or (int, int), optional
        Scalar s or tuple (s_z, s_phi) for convolution striding in z and phi directions.
        If s is a scalar, set (s_z, s_phi) = (s, s).
        In the radial direction, _s_phi_ must evenly divide the original size.
      pad_z: bool, optional
        If True, pad the Z dimension on the input, such that the output size is not reduced when stride_z=1,
        a.k.a. padding="same".
      bias : bool, optional
        Whether or not to include a trainable bias offset.
    
    Shape:
      Input: (*, C_in, Z_in, Phi_in) where * is any number of dimensions including none.
      Output: (*, C_out, Z_out, Phi_out) where for the simplest case of strides=1 and pad_z=False,
        Z_out = Z_in + (k_z-1)
        Phi_out = Phi_in
    '''
    def __init__(self, in_features, out_features, k=(5,3), stride=(1,1), pad_z=False, bias=True):
        super().__init__()
        
        k = _check_arg_pair(k, "kernel size")
        stride = _check_arg_pair(stride, "stride")
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.stride = stride

        if k[0]<stride[0]:
            raise NotImplementedError(f"haven't considered kz<sz yet (got kz={k[0]} sz={stride[0]})")
        self.pad_z_n=0 if not pad_z else (k[0]-1)//2
        self.pad_z_on=0 if not pad_z else stride[0]-1

        if (k[1]>stride[1]):
          self.excess = (k[1]-stride[1])
        elif (k[1]<stride[1]):
            raise NotImplementedError("Haven't implemented this case yet... its equivalent to k=s after padding k with zeros")
        else: # nothing to do when kp=sp, output is already correct shape and no overlap to handle.
            self.excess = 0 

        self.centered = False

        wshape = (in_features, out_features, k[0], k[1])
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(wshape)))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None
    
    def forward(self, x):
        # kz, kp = self.k
        # sz, sp = self.stride

        #https://stackoverflow.com/questions/63508818/how-to-manually-implement-padding-for-pytorch-convolutions
        #https://stackoverflow.com/questions/69782823/understanding-the-pytorch-implementation-of-conv2dtranspose
        # out = (x - 1)s - 2p + d(k - 1) + op + 1
        
        # # NB! We cannot use the builtin bias because we'll break symmetry where the ends wrap around, by adding it twice.
        # x = F.conv_transpose2d(x, self.weights, stride=self.stride)#, padding=(zpad,0))
        
        # if self.pad_z and (kz-sz)>0:
        #     npad = (kz-sz)//2
        #     x = x[...,npad:-npad,:]
        # elif kz<sz:
        #     raise NotImplementedError(f"haven't considered kz<sz yet (got kz={kz} sz={sz})")

        x = F.conv_transpose2d(x, self.weights, stride=self.stride, padding=(self.pad_z_n,0),output_padding=(self.pad_z_on,0))
        
        if (self.excess>0):
            #xa, xb, xc = torch.chunk(x, [excess, -1, excess], dim=-3)
            xa = x[...,:self.excess]
            xb = x[...,self.excess:-self.excess]
            xc = x[...,-self.excess:]
            xac = xa + xc
            if self.centered:
                xc = xac[...,:self.excess//2]
                xa = xac[...,self.excess//2:]
                x = torch.concat([xa, xb, xc], axis=-1)
            else:
                x = torch.concat([xac, xb], axis=-1)
            pass
            
        b = torch.reshape(self.bias, (1,)*(len(x.shape)-3) + (-1,1,1))
        x = x + b
        return x
    
    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, k={self.k}, stride={self.stride}, pad_z={self.pad_z_n}"
