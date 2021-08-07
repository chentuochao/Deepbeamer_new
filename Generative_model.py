import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import asteroid
#------------loss function --------------

class SISDRLoss(nn.Module):
    def __init__(self, offset, l=1e-3):
        super().__init__()
        
        self.l=l
        self.offset=offset
        
    def forward(self, signal, gt):
        return torch.sum(asteroid.losses.pairwise_neg_sisdr(signal[..., self.offset:], gt[..., self.offset:]))+self.l*torch.sum(signal**2)/signal.shape[-1]          
    
class L1Loss(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset=offset
        self.loss=nn.L1Loss()
        
    def forward(self, signal, gt):
        return self.loss(signal[..., self.offset:], gt[..., self.offset:])
    
    
class FuseLoss(nn.Module):
    def __init__(self, offset, r=50):
        super().__init__()
        self.offset=offset
        self.l1loss=nn.L1Loss()
        self.sisdrloss=asteroid.losses.pairwise_neg_sisdr
        self.r=r
        
    def forward(self, signal, gt):
        #print(signal.shape, gt.shape)
        if len(signal.shape)==2:
            signal=signal.unsqueeze(1)
            gt = gt.unsqueeze(1)
        a = signal[..., self.offset:]

        b = gt[..., self.offset:]*self.r + torch.mean(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
        #print(a)
        #print(b)
        #print(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
        #print(self.l1loss(a, b))
        #print("----------------------------")
        return self.l1loss(signal[..., self.offset:], gt[..., self.offset:])*self.r+torch.mean(self.sisdrloss(signal[..., self.offset:], gt[..., self.offset:]))
    


class ComplexSTFTWrapper(nn.Module):
    def __init__(self, win_length, hop_length, center=True):
        super(ComplexSTFTWrapper,self).__init__()
        self.win_length=win_length
        self.hop_length=hop_length
        self.center=center
        
    def transform(self, input_data):
        B,C,L=input_data.shape
        input_data=input_data.view(B*C, L)
        r=torch.stft(input_data, n_fft=self.win_length, hop_length=self.hop_length, center=self.center, onesided=False, return_complex=False)
        _,F,T,_=r.shape
        #print("transform:  ", r.shape)
        r=r.view(B,C,F,T,2)
        return (r[...,0], r[..., 1])
                              
    def reverse(self, input_data):
        r,i=input_data
        B,C,F,T=r.shape
        r=r.flatten(0,1)
        i=i.flatten(0,1)
        input_data=torch.stack([r,i], dim=-1)
        #print("reverse:   ", input_data.shape)
        r=torch.istft(input_data, n_fft=self.win_length, hop_length=self.hop_length, center=self.center, onesided=False,return_complex=False) # B, L
        return r.view(B,C,-1)
        
    def forward(self, x):
        return self.reverse(self.transform(x))


#------------model structure --------------

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class AudioVisual7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual7layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, select_vector):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        select_vector = select_vector.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audioselect_vectorure = torch.cat((select_vector, audio_conv7feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioselect_vectorure)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))
        return mask_prediction

class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 8 + 13, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, select_vector):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        
        #print(select_vector.size(), audio_conv1feature.size(),audio_conv2feature.size(), audio_conv3feature.size(), audio_conv4feature.size(), audio_conv5feature.size())
        select_vector = torch.unsqueeze(select_vector, -1)
        select_vector = torch.unsqueeze(select_vector, -1)

        select_vector = select_vector.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])
        #print(select_vector.size())
        audioVisual_feature = torch.cat((select_vector, audio_conv5feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        #print(audio_upconv1feature.size(), audio_conv4feature.size(),  audio_upconv1feature.size() )
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction


class Earbud_Net(nn.Module):
    def __init__(self, block_size = 64, unet_num_layers=5, ngf=64, input_nc=1, output_nc=1, weights=''):
        super(Earbud_Net, self).__init__()
        self.stft=ComplexSTFTWrapper(hop_length=block_size//2, win_length=block_size*2)
        if unet_num_layers == 7:
            self.net = AudioVisual7layerUNet(ngf, input_nc, output_nc)
        elif unet_num_layers == 5:
            self.net = AudioVisual5layerUNet(ngf, input_nc, output_nc)

        self.net.apply(weights_init)

        if len(weights) > 0:
            print('Loading weights for UNet')
            self.net.load_state_dict(torch.load(weights))

    def forward(self, wav, select_vector):
        #print(wav.dtype)
        rea, imag=self.stft.transform(wav) #B,C,F,T, the spectrum process
        B,Cin,_,T=rea.shape
        #print(wav.shape, rea.shape)
        mag = torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2)))
        pha = torch.atan2(imag.data, rea.data)
        #print(mag.size(),select_vector.size(),  mag.dtype, select_vector.dtype  )
        mask = self.net(mag, select_vector)
        #print(mask.size())
        pred_mag = mask*mag
        pred_real = pred_mag * torch.cos(pha)
        pred_image = pred_mag * torch.sin(pha)

        pred_audio = self.stft.reverse((pred_real, pred_image))
        #print("pred_audio: ", pred_audio.shape)
        return pred_audio



