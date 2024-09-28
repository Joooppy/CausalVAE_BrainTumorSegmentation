"""
Base NvNet partly adapted from:
@author: Chenggang
@github: https://github.com/MissShihongHowRU

CausalVAE partly adapted from:
@author: Yang
@paper: https://arxiv.org/abs/2004.08697.pdf

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cvae_utils as ut
import numpy as np
from mask import MaskLayer, DagLayer, Attention
from custom_modules import DownSampling, EncoderBlock, ConvEncoder, LinearUpSampling, DecoderBlock, OutputTransition, Decoder_DAG
from collections.abc import Iterable

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")


class CausalVAE(nn.Module):
    def __init__(self, nn='mask', name='vae', z_dim=9, z1_dim=3, z2_dim=3, inference = False, initial=True):
        super().__init__()
        self.name = name
        self.z_dim = z_dim 
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim 
        self.channel = 4
        self.scale = np.array([[20,15],[2,2],[59.5,26.5], [10.5, 4.5]])

        self.enc = ConvEncoder() 
        self.dec = Decoder_DAG(self.z_dim,self.z1_dim, self.z2_dim)
        self.attn = Attention(self.z2_dim)
        self.dag = DagLayer(self.z1_dim, self.z1_dim, i=inference, initial=initial)
        
        self.mask_z = MaskLayer(self.z_dim, concept=self.z1_dim, z2_dim=self.z2_dim)
        self.mask_u = MaskLayer(self.z1_dim, concept=self.z1_dim, z2_dim=1)


    def forward(self, x, label, mask=None, sample=False, adj=None, lambdav=0.001):
        if label is None:
            raise ValueError("Label tensor is None in CausalVAE forward method")
        nelbo, kl, rec, decoded_logits, z_given_dag = self.negative_elbo_bound(x, label, mask, sample, adj, lambdav)
        
        summaries = {
            'train/loss': nelbo,
            'gen/elbo': -nelbo,
            'gen/kl_z': kl,
            'gen/rec': rec,
        }
        return nelbo, kl, summaries, decoded_logits, z_given_dag
    
    def negative_elbo_bound(self, x, x_input, label, mask = None, sample = False, adj = None, lambdav=0.001):
        """
            Computes the Evidence Lower Bound, KL and, Reconstruction costs

            Args:
                x: tensor: (batch, dim): Observations

            Returns:
                nelbo: tensor: (): Negative evidence lower bound
                kl: tensor: (): ELBO KL divergence to prior
                rec: tensor: (): ELBO Reconstruction term
            """
        print("Starting negative_elbo_bound...")
        if label is None:
            raise ValueError("Label tensor is None")
        if x is None:
            raise ValueError("Input tensor x is None")
        
        print("label dimension Correct? ", label.size()[1] == self.z1_dim)

        assert label.size()[1] == self.z1_dim

        # initialize gaussian distribution from encoded input, calculate mean and  set variance to uniform distribution
        #q_m, q_v = ut.gaussian_parameters(x.to(device), dim=1)
        q_m, q_v = self.enc.encode(x.to(device))

        q_m, q_v = q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device)

        
        # create DAG and adjust to z dimensions
        decode_m, decode_v = self.dag.calculate_dag(q_m.to(device), torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device))
        decode_m, decode_v = decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), decode_v

        # learn DAG with masks (z) and attention, set u-mask to labels
        if not sample:
            if isinstance(mask, Iterable):  # Check if mask is iterable
                for label_index in mask:
                    z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                    decode_m[:, label_index, :] = z_mask[:, label_index, :]
                    decode_v[:, label_index, :] = z_mask[:, label_index, :]
            else:
                print("mask is not iterable:", mask)
            
            #if mask is not None and mask in [0, 1, 3]:
            #    z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
            #    decode_m[:, mask, :] = z_mask[:, mask, :]
            #    decode_v[:, mask, :] = z_mask[:, mask, :]

            m_zm, m_zv = self.dag.mask_z(decode_m.to(device)).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]), decode_v.reshape([q_m.size()[0], self.z1_dim, self.z2_dim])
            m_u = self.dag.mask_u(label.to(device))

            f_z = self.mask_z.mix(m_zm).reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)
            

            e_tilde = self.attn.attention(decode_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device), q_m.reshape([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device))[0]
            f_z1 = f_z + e_tilde

            if isinstance(mask, Iterable):
                for label_index in mask:
                    z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
                    f_z1[:, label_index, :] = z_mask[:, label_index, :]
                    m_zv[:, label_index, :] = z_mask[:, label_index, :]
            else:
                print("mask is not iterable:", mask)
            #if mask is not None and mask == 2:
            #    z_mask = torch.ones(q_m.size()[0], self.z1_dim, self.z2_dim).to(device) * adj
            #    f_z1[:, mask, :] = z_mask[:, mask, :]
            #    m_zv[:, mask, :] = z_mask[:, mask, :]

            g_u = self.mask_u.mix(m_u).to(device)
            m_zv = torch.ones([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)

            z_given_dag = ut.conditional_sample_gaussian(f_z1, q_v * lambdav)

        # decode and calculate benoulli logits from z after DAG 
        decoded_bernoulli_logits, x1, x2, x3, x4 = self.dec.decode_sep(z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(device))
        
        # calculate reconstruction
        rec = ut.log_bernoulli_with_logits(x_input, decoded_bernoulli_logits.reshape(x_input.size()))
        rec = -torch.mean(rec)  # negative to ensure minimizing this loss
        # set priors and calculate conditional priors
        p_m, p_v = torch.zeros(q_m.size()), torch.ones(q_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)

        # sample conditional prior
        cp_v = torch.ones([q_m.size()[0], self.z1_dim, self.z2_dim]).to(device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(device), cp_v.to(device))
        
        # initialize KL divergence and calculate with divergence betweend aproximate posterior and standard normal prior
        kl = torch.zeros(1).to(device)
        kl = 0.3 * ut.kl_normal(q_m.view(-1, self.z_dim).to(device), q_v.view(-1, self.z_dim).to(device), p_m.view(-1, self.z_dim).to(device), p_v.view(-1, self.z_dim).to(device))

        # loop over dimensions and calculate mean
        for i in range(self.z1_dim):
            kl = kl + 1 * ut.kl_normal(decode_m[:, i, :].to(device), cp_v[:, i, :].to(device), cp_m[:, i, :].to(device), cp_v[:, i, :].to(device))
        kl = torch.mean(kl)

        # initialize KL for mask and loop over concepts
        mask_kl = torch.zeros(1).to(device)
        mask_kl2 = torch.zeros(1).to(device)

        for i in range(self.z1_dim):
            mask_kl = mask_kl + 1 * ut.kl_normal(f_z1[:, i, :].to(device), cp_v[:, i, :].to(device), cp_m[:, i, :].to(device), cp_v[:, i, :].to(device))
        # calculate mask loss of KL divergence with mse between mask output and labels 
        u_loss = torch.nn.MSELoss()
        mask_l = torch.mean(mask_kl) + u_loss(g_u, label.float().to(device))
        
        # negative evidence lower bout, final loss consisting of reconstruction loss, KL and masked loss
        nelbo = rec + kl + mask_l
        print("nelbo: ", nelbo)

        # returns nelbo, kl, reconstuction loss, decoded logits and z after DAG
        return nelbo, kl, rec, decoded_bernoulli_logits.reshape(x_input.size()), z_given_dag
    

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))

class NvNet(nn.Module):
    def __init__(self, config):
        super(NvNet, self).__init__()
        
        self.config = config
        # some critical parameters
        self.inChans = config["input_shape"][1]
        self.input_shape = config["input_shape"]
        self.seg_outChans = 3
        self.activation = config["activation"]
        self.normalization = config["normalization"]
        self.mode = config["mode"]
        
        # Encoder Blocks
        self.in_conv0 = DownSampling(inChans=self.inChans, outChans=32, stride=1, dropout_rate=0.2)
        self.en_block0 = EncoderBlock(32, 32, activation=self.activation, normalization=self.normalization)
        self.en_down1 = DownSampling(32, 64)
        self.en_block1_0 = EncoderBlock(64, 64, activation=self.activation, normalization=self.normalization)
        self.en_block1_1 = EncoderBlock(64, 64, activation=self.activation, normalization=self.normalization)
        self.en_down2 = DownSampling(64, 128)
        self.en_block2_0 = EncoderBlock(128, 128, activation=self.activation, normalization=self.normalization)
        self.en_block2_1 = EncoderBlock(128, 128, activation=self.activation, normalization=self.normalization)
        self.en_down3 = DownSampling(128, 256)
        self.en_block3_0 = EncoderBlock(256, 256, activation=self.activation, normalization=self.normalization)
        self.en_block3_1 = EncoderBlock(256, 256, activation=self.activation, normalization=self.normalization)
        self.en_block3_2 = EncoderBlock(256, 256, activation=self.activation, normalization=self.normalization)
        self.en_block3_3 = EncoderBlock(256, 256, activation=self.activation, normalization=self.normalization)
        
        # Decoder Blocks
        self.de_up2 = LinearUpSampling(256, 128, mode=self.mode)
        self.de_block2 = DecoderBlock(128, 128, activation=self.activation, normalization=self.normalization)
        self.de_up1 = LinearUpSampling(128, 64, mode=self.mode)
        self.de_block1 = DecoderBlock(64, 64, activation=self.activation, normalization=self.normalization)
        self.de_up0 = LinearUpSampling(64, 32, mode=self.mode)
        self.de_block0 = DecoderBlock(32, 32, activation=self.activation, normalization=self.normalization)
        self.de_end = OutputTransition(32, self.seg_outChans)
        
        # Causal VAE
        if self.config["CausalVAE_enable"]:
            self.dense_features = (self.input_shape[2]//16, self.input_shape[3]//16, self.input_shape[4]//16)  # 8, 12, 12
            self.causal_vae = CausalVAE(nn='mask', name='causal_vae', z_dim=9, z1_dim=3, z2_dim=3)

    def forward(self, x, targets, label_distr):
        print("Initial input size:", x.size())
        x_input = x.clone().detach()
        out_init = self.in_conv0(x)
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        out_en3 = self.en_block3_3(
            self.en_block3_2(
                self.en_block3_1(
                    self.en_block3_0(
                        self.en_down3(out_en2)))))

        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))
        out_end = self.de_end(out_de0)
        print("After de_end size:", out_end.size())

        if self.config["CausalVAE_enable"] and targets is not None:
            #z = self.cvae_encoder(out_en3)

            nelbo, kl, summaries, rec_image, z_given_dag = self.causal_vae(out_en3, x_input, label_distr)
            out_final = torch.cat((out_end, rec_image), 1)
            return out_final, kl, nelbo, summaries, rec_image, z_given_dag

        return out_end
