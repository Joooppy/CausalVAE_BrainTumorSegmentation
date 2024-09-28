import torch
import torch.nn as nn
import torch.nn.functional as F
from siren_pytorch import Sine
import numpy as np

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

class DownSampling(nn.Module):
    # 3x3x3 convolution and 1 padding as default
    def __init__(self, inChans, outChans, stride=2, kernel_size=3, padding=1, dropout_rate=None):
        super(DownSampling, self).__init__()
        
        self.dropout_flag = False
        self.conv1 = nn.Conv3d(in_channels=inChans, 
                     out_channels=outChans, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding,
                     bias=False)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate,inplace=True)
            
    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out
    
class EncoderBlock(nn.Module):
    '''
    Encoder block; Green
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalization="group_normalization"):
        super(EncoderBlock, self).__init__()
        
        if normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out


class ConvEncoder(nn.Module):
    def __init__(self, out_dim=9):
        super().__init__()
        # Initial tensor shape [batch_size, 256, 16, 24, 20]

        # Convolutional layers adjusted for 3D input with example strides and paddings
        self.convm = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        self.convv = nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)
        
        # Flatten the output dimensions for the linear layers
        flattened_size = 128 * 8 * 12 * 10  # Channels * Depth * Height * Width after convolutions

        # Linear layers to produce mean and variance
        self.mean_layer = nn.Linear(flattened_size, out_dim)
        self.var_layer = nn.Linear(flattened_size, out_dim)

    def encode(self, x):
        # Applying the convolutional layers
        hm = self.convm(x)
        hv = self.convv(x)

        # Flatten the outputs for the linear layers
        hm = hm.view(hm.size(0), -1)  # Flatten keeping the batch dimension
        hv = hv.view(hv.size(0), -1)

        # Generate mean and variance
        mu = self.mean_layer(hm)
        var = self.var_layer(hv)
        var = F.softplus(var) + 1e-8

        # Reshape accordingly
        mu = mu.view(-1, 3, 3)
        var = var.view(-1, 3, 3)
        
        return mu, var


    
class LinearUpSampling(nn.Module):
    '''
    Trilinear interpolate to upsampling
    '''
    def __init__(self, inChans, outChans, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        # self.conv2 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
    
    def forward(self, x, skipx=None):
        out = self.conv1(x)
        # out = self.up1(out)
        out = nn.functional.interpolate(out, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        
        if skipx is not None:
            out += skipx
            # out = torch.cat((out, skipx), 1)
            # out = self.conv2(out)  # Given groups=1, weight of size [128, 256, 1, 1, 1], expected input[1, 128, 32, 48, 40] to have 256 channels, but got 128 channels instead


        return out
    
class DecoderBlock(nn.Module):
    '''
    Decoder block
    '''
    def __init__(self, inChans, outChans, stride=1, padding=1, num_groups=8, activation="relu", normalization="group_normalization"):
        super(DecoderBlock, self).__init__()
        
        num_groups = outChans if outChans % 8 == 0 else 1
        
        if normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=outChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=outChans, out_channels=outChans, kernel_size=3, stride=stride, padding=padding)
        
        
    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        
        out += residual
        
        return out
    
class OutputTransition(nn.Module):
    '''
    Decoder output layer 
    output the prediction of segmentation result
    '''
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=outChans, kernel_size=1)
        self.actv1 = torch.sigmoid
        
    def forward(self, x):
        return self.actv1(self.conv1(x))

class VDResampling(nn.Module):
    '''
    Variational Auto-Encoder Resampling block
    '''
    def __init__(self, inChans=256, outChans=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1,
                 activation="relu", normalization="group_normalization"):
        super(VDResampling, self).__init__()
        
        if normalization == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=8, num_channels=inChans)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=16*dense_features[0]*dense_features[1]*dense_features[2], out_features=256)
        self.dense2 = nn.Linear(in_features=128, out_features=128*dense_features[0]*dense_features[1]*dense_features[2])
        self.up0 = LinearUpSampling(128, outChans)
        
    def forward(self, x):
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)   # 16*10*12*8  # 16, 8, 12, 12
        out = out.view(-1, self.num_flat_features(out))  # flatten  16*8*12*12
        out_vd = self.dense1(out)
        distr = out_vd 
        out = VDraw(out_vd)  # 128
        out = self.dense2(out)
        out = self.actv2(out)
        out = out.view((-1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))  # flat to conv
        # out = out.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)  # include conv1 and upsize 256*20*24*16
        
        return out, distr
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features

def VDraw(x):
    loc = x[:, :128]
    scale = x[:, 128:]
    scale = F.softplus(scale)  # Apply softplus to ensure the scale is positive
    return torch.distributions.Normal(loc, scale).sample()

class VDecoderBlock(nn.Module):
    '''
    Variational Decoder block
    '''
    def __init__(self, inChans, outChans, activation="relu", normalization="group_normalization", mode="trilinear"):
        super(VDecoderBlock, self).__init__()

        self.up0 = LinearUpSampling(inChans, outChans, mode=mode)
        self.block = DecoderBlock(outChans, outChans, activation=activation, normalization=normalization)
    
    def forward(self, x):
        out = self.up0(x)
        out = self.block(out)

        return out

#class VAE(nn.Module):
    '''
    Variational Auto-Encoder : to group the features extracted by Encoder
##    '''
#    def __init__(self, inChans=256, outChans=4, dense_features=(10, 12, 8),
#                 activation="relu", normalization="group_normalization", mode="trilinear"):
#        super(VAE, self).__init__()
#
#        self.vd_resample = VDResampling(inChans=inChans, outChans=inChans, dense_features=dense_features)
#        self.vd_block2 = VDecoderBlock(inChans, inChans//2)
#        self.vd_block1 = VDecoderBlock(inChans//2, inChans//4)
#        self.vd_block0 = VDecoderBlock(inChans//4, inChans//8)
#        self.vd_end = nn.Conv3d(inChans//8, outChans, kernel_size=1)
#        
#    def forward(self, x):
#        out, distr = self.vd_resample(x)
#        out = self.vd_block2(out)
#        out = self.vd_block1(out)
#       out = self.vd_block0(out)
#        out = self.vd_end(out)
#
#        return out, distr

class CVAE_Encoder(nn.Module):
    def __init__(self, out_dim=16, inChans=256, z_dim=16):
        super(CVAE_Encoder, self).__init__()
        
        #self.conv1 = nn.Conv3d(inChans, 128, kernel_size=3, stride=2, padding=1)  # (128, 8, 12, 12)
        #self.conv2 = nn.Conv3d(128, 64, kernel_size=3, stride=2, padding=1)       # (64, 4, 6, 6)
        #self.conv3 = nn.Conv3d(64, 32, kernel_size=3, stride=2, padding=1)        # (32, 2, 3, 3)
        #self.conv4 = nn.Conv3d(32, 16, kernel_size=3, stride=2, padding=1)        # (16, 1, 2, 2)
        
        #flat_features = 16*2*2*1
        
        #self.fc = nn.Linear(flat_features, out_dim)

        self.LReLU = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        #print('Input size:', x.size())
        #x = self.LReLU(self.conv1(x))
        #print('After conv1 size:', x.size())
        #x = self.LReLU(self.conv2(x))
        #print('After conv2 size:', x.size())
        #x = self.LReLU(self.conv3(x))
        #print('After conv3 size:', x.size())
        #x = self.LReLU(self.conv4(x))
        #print('After conv4 size:', x.size())

        
        x = x.view(x.size(0), -1)  # Flatten
        
        #x = self.fc(x)
        #print('Output size:', x.size())
        return x
    
class zResampling(nn.Module):
    '''
    z upsampling block to fit Conv3D dimensions
    '''
    def __init__(self, inChans=4, outChans=256, dense_features=(10, 12, 8), stride=2, kernel_size=3, padding=1,
                 activation="relu", normalization="group_normalization"):
        super(zResampling, self).__init__()
        
        if normalization == "group_normalization":
            self.gn1 = nn.GroupNorm(num_groups=16, num_channels=32)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        # elif activation == "elu":
        #     self.actv1 = nn.ELU(inplace=True)
        #     self.actv2 = nn.ELU(inplace=True)
        elif activation == "sin":
            self.actv1 = Sine(1.0)
            self.actv2 = Sine(1.0)
        self.dense_features = dense_features
        
        self.conv1 = nn.Conv3d(in_channels=inChans, out_channels=16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.dense1 = nn.Linear(in_features=3, out_features=16)
        self.dense2 = nn.Linear(in_features=16, out_features=128)
        self.dense3 = nn.Linear(in_features=inChans*4, out_features=inChans*8)
        self.dense4 = nn.Linear(in_features=inChans*8, out_features=inChans*16)
        self.dense5 = nn.Linear(in_features=128, out_features=128*dense_features[0]*dense_features[1]*dense_features[2])
        self.up0 = LinearUpSampling(128, outChans)
        
        
    def forward(self, x):
        out = self.dense1(x)
        out = self.actv2(out)
        out = self.dense2(out)
        out = self.actv2(out)
        out = self.dense5(out)
        out = self.actv2(out)
        out = out.view((-1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))  # flat to conv
        # out = out.view((1, 128, self.dense_features[0], self.dense_features[1], self.dense_features[2]))
        out = self.up0(out)  # include conv1 and upsize 256*20*24*16
        return out

class Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel = 4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.decode_channels = 256
        self.z1_dim = z1_dim
        self.concept = concept #categories in latent space
        self.y_dim = y_dim # dimensionality of additional labels
        self.channel = channel # output channels
        #print(self.channel)
        
        # VDecoderBlocks to decode according to image task
        self.elu = nn.ELU()        
        self.vd_block2 = VDecoderBlock(self.decode_channels, self.decode_channels//2)  # Adjust z_dim for use case
        self.vd_block1 = VDecoderBlock(self.decode_channels//2, self.decode_channels//4)
        self.vd_block0 = VDecoderBlock(self.decode_channels//4, self.decode_channels//8)
        self.vd_end = nn.Conv3d(self.decode_channels//8, channel, kernel_size=1)
        self.z_resample = zResampling()
        
    def decode_condition(self, z, u):
        #z = z.view(-1,3*4)
        z = z.view(-1, 3*4)
        z1, z2, z3 = torch.split(z, self.z_dim//4, dim = 1)
        #print(u[:,0].reshape(1,u.size()[0]).size())
        rx1 = self.net1(torch.transpose(torch.cat((torch.transpose(z1, 1,0), u[:,0].reshape(1,u.size()[0])), dim = 0), 1, 0))
        rx2 = self.net2(torch.transpose(torch.cat((torch.transpose(z2, 1,0), u[:,1].reshape(1,u.size()[0])), dim = 0), 1, 0))
        rx3 = self.net3(torch.transpose(torch.cat((torch.transpose(z3, 1,0), u[:,2].reshape(1,u.size()[0])), dim = 0), 1, 0))
   
        h = self.net4(torch.cat((rx1, rx2, rx3), dim=1))
        return h

    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            elif self.concept == 3:
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
            elif self.concept == 3:
                zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)

        # apply vdecoderblocks to adjust to convolutional model and reconstructed image dim
        zy1 = self.z_resample(zy1)
        zy2 = self.z_resample(zy2)
        zy3 = self.z_resample(zy3)

        rx1 = self.vd_end(self.vd_block0(self.vd_block1(self.vd_block2(zy1))))
        rx2 = self.vd_end(self.vd_block0(self.vd_block1(self.vd_block2(zy2))))
        rx3 = self.vd_end(self.vd_block0(self.vd_block1(self.vd_block2(zy3))))
        # Variation for number of concepts
        if self.concept == 4:
            zy4 = self.z_resample(zy4)
            rx4 = self.vd_end(self.vd_block0(self.vd_block1(self.vd_block2(zy4))))
            h = (rx1 + rx2 + rx3 + rx4) / self.concept
        elif self.concept == 3:
            h = (rx1 + rx2 + rx3) / self.concept
        return h, h, h, h, h
    

