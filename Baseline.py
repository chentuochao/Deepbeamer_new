import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
import numpy as np

def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]
    #print("out:", outer_dimensions)
    #print(frames, frame_length )
    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)
    frame = frame.cuda(0)
    #print(frame.device, subframe_signal.device)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

class Base_Model(nn.Module):
    def __init__(self, L, N, B, num_class):
        super(Base_Model, self).__init__()
        self.L, self.N, self.B =  L, N, B
        self.encoder = nn.Sequential(
            nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False),
            nn.ReLU(True)
        )
        self.network = nn.Sequential(
            nn.Conv1d(N + num_class, B, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(B, N, 1, bias=False),
        )
        self.act = nn.ReLU()
        self.decoder = nn.Linear(N, L, bias=False)
        

    def forward(self, mixture, label_vector):
        mixture_w = self.encoder(mixture)

        M, N, K = mixture_w.size()
        label_vector = torch.unsqueeze(label_vector, 2)
        #print(mixture_w.size())
        label_vector = label_vector.repeat(1, 1, K)
        features = torch.cat((label_vector, mixture_w), dim=1)

        score = self.network(features)  # [M, N, K] -> [M, N, K]
        score = score.view(M, 1, N, K) # [M, C*N, K] -> [M, C, N, K]
        source_w = self.act(score)

        source_w = torch.transpose(source_w, 2, 3) # [M, C, K, N]
        est_source = self.decoder(source_w)  # [M, C, K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T

        return est_source

if __name__ == "__main__":
    torch.manual_seed(123)
    M, N, L, T, B = 15, 32, 128, 12800+64, 64
    Class_num = 3
    temp_labels = np.zeros((M, Class_num))
    temp_labels[1] = 1
    temp_labels = torch.from_numpy(temp_labels)
    temp_labels = temp_labels.to(torch.float32)
    #temp_labels = temp_labels.astype("double")
    K = 2*T//L-1#512

    mixture = torch.rand(M, 1, T)
    
    # test Conv-TasNet
    conv_tasnet = Base_Model(L, N, B, Class_num)
    est_source = conv_tasnet(mixture, temp_labels)
    print('est_source', est_source)
    print('est_source size', est_source.size())
