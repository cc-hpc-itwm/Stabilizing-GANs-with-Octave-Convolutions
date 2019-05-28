from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torch.nn.functional as F
import numpy as np
from octave import *


class DCGAN(nn.Module):
    """
    Implementation of DCGAN
    'Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks'
    A. Radford, L. Metz & S. Chintala
    arXiv:1511.06434v2

    This is only a container for the generator/discriminator architecture
    and weight initialization schemes. No optimizers are attached.
    """

    def __init__(self, gan_type='gan', latent_dim=100, batch_size=64,
            use_cuda=True):
        super(DCGAN, self).__init__()
        self.gan_type = gan_type
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.G = Generator()
        self.D = Discriminator()
        self.init_weights(self.G)
        self.init_weights(self.D)

        self.y_real = Variable(torch.ones(batch_size))
        self.y_fake = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available() and self.use_cuda:
            self.y_real = self.y_real.cuda()
            self.y_fake = self.y_fake.cuda()


    def load_model(self, filename, use_cuda=True):
        """Load PyTorch model"""

        print('Loading generator checkpoint from: {}'.format(filename))
        if use_cuda:
            self.G.load_state_dict(torch.load(filename))
        else:
            self.G.load_state_dict(torch.load(filename, map_location='cpu'))


    def save_model(self, ckpt_path, epoch, override=True):
        """Save model"""

        if override:
            fname_gen_pt = '{}/{}-gen.pt'.format(ckpt_path, self.gan_type)
            fname_disc_pt = '{}/{}-disc.pt'.format(ckpt_path, self.gan_type)
        else:
            fname_gen_pt = '{}/{}-gen-epoch-{}.pt'.format(ckpt_path, self.gan_type, epoch + 1)
            fname_disc_pt = '{}/{}-disc-epoch-{}.pt'.format(ckpt_path, self.gan_type, epoch + 1)

        print('Saving generator checkpoint to: {}'.format(fname_gen_pt))
        torch.save(self.G.state_dict(), fname_gen_pt)
        sep = '\n' + 80 * '-'
        print('Saving discriminator checkpoint to: {}{}'.format(fname_disc_pt, sep))
        torch.save(self.D.state_dict(), fname_disc_pt)


    def init_weights(self, model):
        """Initialize weights and biases (according to paper)"""

        for m in model.parameters():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def get_num_params(self):
        """Compute the total number of parameters in model"""

        num_params_D, num_params_G = 0, 0
        for p in self.D.parameters():
            num_params_D += p.data.view(-1).size(0)
        for p in self.G.parameters():
            num_params_G += p.data.view(-1).size(0)
        return num_params_D, num_params_G


    def create_latent_var(self, batch_size, seed=None):
        """Create latent variable z"""

        if seed:
            torch.manual_seed(seed)
        z = Variable(torch.randn(batch_size, self.latent_dim))
        if torch.cuda.is_available() and self.use_cuda:
            z = z.cuda()
        return z


    def train_G(self, freq_weight, G_optimizer, batch_size):
        """Update generator parameters"""
        self.G.zero_grad()
        self.D.zero_grad()

        # Through generator, then discriminator
        z = self.create_latent_var(self.batch_size)
        fake_imgs = self.G(z, freq_weight)
        D_out_fake = self.D(fake_imgs, freq_weight)

        if self.gan_type == 'gan':
            # Evaluate loss and backpropagate
            G_train_loss = F.binary_cross_entropy_with_logits(D_out_fake, self.y_real)
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.item()

        elif self.gan_type == 'lsgan':
            # Evaluate loss and backpropagate (negative since we minimize)
            G_train_loss = torch.mean((D_out_fake - 1) ** 2)
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.item()

        elif self.gan_type == 'wgan':
            # Evaluate loss and backpropagate (negative since we minimize)
            G_train_loss = -D_out_fake.mean()
            G_train_loss.backward()
            G_optimizer.step()

            #  Update generator loss
            G_loss = G_train_loss.item()


        else:
            raise NotImplementedError

        return G_loss


    def train_D(self, x, freq_weight, D_optimizer, batch_size):
        """Update discriminator parameters"""

        self.D.zero_grad()

        # Through generator, then discriminator
        D_out_real = self.D(x, freq_weight)
        z = self.create_latent_var(self.batch_size)
        fake_imgs = self.G(z, freq_weight).detach()
        D_out_fake = self.D(fake_imgs, freq_weight)

        if self.gan_type == 'gan':
            D_real_loss = F.binary_cross_entropy_with_logits(D_out_real, self.y_real)
            D_fake_loss = F.binary_cross_entropy_with_logits(D_out_fake, self.y_fake)

            # Update discriminator
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

        elif self.gan_type == 'lsgan':
            # Update discriminator
            D_real_loss = torch.mean((D_out_real - 1) ** 2)
            D_fake_loss = torch.mean(D_out_fake ** 2)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

        elif self.gan_type == 'wgan':
            # Update discriminator (negative since we minimize)
            D_train_loss = -(D_out_real.mean() - D_out_fake.mean())
            D_train_loss.backward()
            D_optimizer.step()

            # Clip weights
            self.D.clip()

        else:
            raise NotImplementedError

        # Update discriminator loss
        D_loss = D_train_loss.item()
        return D_loss, fake_imgs


    def generate_img(self, z=None, n=1, seed=None):
        """Sample random image from GAN"""
        # Nothing was provided, sample
        if z is None and seed is None:
            z = self.create_latent_var(n)
        # Seed was provided, use it to sample
        elif z is None and seed is not None:
            z = self.create_latent_var(n, seed)
        return self.G(z, 0.5)#.squeeze()

#---------------------
#---------------------
#------CELEBA-128-----
#---------------------
#---------------------
# Octave Disc with decay variable (celeba 128)
class Discriminator(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self):
        super(Discriminator, self).__init__()
        
        d=32
        self.l0 = FirstOctaveCBL(3, d, kernel_size=3, alpha=0.5, stride=2, padding=1)
        self.l1 = OctaveCBL(d, d*2, kernel_size=3, alpha=0.5, stride=2, padding=1)
        self.l2 = OctaveCBL(d*2, d*4, kernel_size=3, alpha=0.5, stride=2, padding=1)
        self.l3 = OctaveCBL(d*4, d*8, kernel_size=3, alpha=0.5, stride=2, padding=1)
        self.l4 = OctaveCBL(d*8, d*16, kernel_size=3, alpha=0.5, stride=2, padding=1)
        self.l5 = LastOctaveConv_(d*16, 1, kernel_size=3, alpha=0.5, stride=2, padding=1)
        

    def forward(self, x, freq_weight):
        x = self.weighting(self.l0(x), freq_weight)
        x = self.weighting(self.l1(x), freq_weight)
        x = self.weighting(self.l2(x), freq_weight)
        x = self.weighting(self.l3(x), freq_weight)
        x = self.weighting(self.l4(x), freq_weight)
        x = self.l5(x).view(-1)
        return x

    def weighting(self, x, freq_weight):
        
        if freq_weight > 0.5:
            freq_weight=0.7
        else:
            freq_weight+=0.2
            
        x_h, x_l = x
        x_h = freq_weight*x_h
        x_l = (1-freq_weight)*x_l
        x = x_h, x_l
        return x

    def clip(self, c=0.05):
        """Weight clipping in (-c, c)"""

        for p in self.parameters():
            p.data.clamp_(-c, c)


#Octave Gen with decay (celeba 128)
class Generator(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
    
        
        # Project and reshape
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True))

        d = 32
        #nn.ReflectionPad2d(1)
        self.l0 = FirstOctaveCBR(d*16, d*8, kernel_size=3, alpha=0.5, stride=1, padding=1)
        self.l1 = OctaveCBR(d*8, d*4, kernel_size=3, alpha=0.5, stride=1, padding=1)
        self.l2 = OctaveCBR(d*4, d*2, kernel_size=3, alpha=0.5, stride=1, padding=1)
        self.l3 = OctaveCBR(d*2, d, kernel_size=3, alpha=0.5, stride=1, padding=1)
        self.l4 = LastOCtaveUp(d, 3, kernel_size=3, alpha=0.5, stride=1, padding=1)
        self.tanh = nn.Tanh()


    def forward(self, x, freq_weight):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        x = self.weighting(self.l0(x), freq_weight)
        x = self.weighting(self.l1(x), freq_weight)
        x = self.weighting(self.l2(x), freq_weight)
        x = self.weighting(self.l3(x), freq_weight)
        x = self.tanh(self.l4(x))
        return x
    
    def weighting(self, x, freq_weight):
                
        if freq_weight > 0.5:
            freq_weight=0.7
        else:
            freq_weight+=0.2
            
        x_h, x_l = x
        x_h = freq_weight*x_h
        x_l = (1-freq_weight)*x_l
        x = x_h, x_l
        return x