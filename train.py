from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dcgan import DCGAN, Generator, Discriminator
import torchvision.utils

import numpy as np
import pickle
import glob, os, sys
import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import utils
import subprocess as sp
import argparse

import torchvision.transforms as transforms


class CelebA(object):
    """Implement DCGAN for CelebA dataset"""

    def __init__(self, train_params, ckpt_params, gan_params):
        # Training parameters
        self.root_dir = train_params['root_dir']
        self.batch_size = train_params['batch_size']
        self.train_len = train_params['train_len']
        self.learning_rate = train_params['learning_rate']
        self.momentum = train_params['momentum']
        self.optim = train_params['optim']
        self.use_cuda = train_params['use_cuda']

        # Checkpoint parameters (when, where)
        self.batch_report_interval = ckpt_params['batch_report_interval']
        self.ckpt_path = ckpt_params['ckpt_path']
        self.save_stats_interval = ckpt_params['save_stats_interval']

        # Create directories if they don't exist
        if not os.path.isdir(self.ckpt_path):
            print(self.ckpt_path)
            os.mkdir(self.ckpt_path)

        # GAN parameters
        self.gan_type = gan_params['gan_type']
        self.latent_dim = gan_params['latent_dim']
        self.n_critic = gan_params['n_critic']

        # Make sure report interval divides total num of batches
        self.num_batches = self.train_len // self.batch_size

        self.compile()
        #frequency weight
        self.freq_weight = 0


    def compile(self):
        """Compile model (loss function, optimizers, etc.)"""

        # Create new GAN
        self.gan = DCGAN(self.gan_type, self.latent_dim, self.batch_size, self.use_cuda)

        # Set optimizers for generator and discriminator
        if self.optim == 'adam':
            self.G_optimizer = optim.Adam(self.gan.G.parameters(),
                lr=self.learning_rate,
                betas=self.momentum)
            self.D_optimizer = optim.Adam(self.gan.D.parameters(),
                lr=self.learning_rate,
                betas=self.momentum)

        elif self.optim == 'rmsprop':
            self.G_optimizer = optim.RMSprop(self.gan.G.parameters(),
                lr=self.learning_rate)
            self.D_optimizer = optim.RMSprop(self.gan.D.parameters(),
                lr=self.learning_rate)

        else:
            raise NotImplementedError

        # CUDA support
        if torch.cuda.is_available() and self.use_cuda:
            self.gan = self.gan.cuda()

    def save_stats(self, stats):
        """Save model statistics"""

        fname_pkl = '{}/{}-stats.pkl'.format(self.ckpt_path, self.gan_type)
        print('Saving model statistics to: {}'.format(fname_pkl))
        with open(fname_pkl, 'wb') as fp:
            pickle.dump(stats, fp)

    def test(self, epoch):
        fname_gen_pt = '{}/{}-gen-epoch-{}.pt'.format(self.ckpt_path, self.gan_type, epoch + 1)
        self.gan.load_model(fname_gen_pt)
        
        directory = self.ckpt_path+"/testing/"+str(epoch + 1)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Evaluation mode
        self.gan.G.eval()
        n = 10000
        # Predict images to see progress
        for i in range(n):
            img = self.gan.generate_img()
            img = utils.unnormalize(img.squeeze())
            fname_in = '{}/{:d}_test.png'.format(directory, i)
            torchvision.utils.save_image(img, fname_in)
            

    def train(self, nb_epochs, data_loader):
        """Train model on data"""

        # Initialize tracked quantities and prepare everything
        G_all_losses, D_all_losses, times = [], [], utils.AvgMeter()
        utils.format_hdr(self.gan, self.root_dir, self.train_len)
        start = datetime.datetime.now()

        g_iter, d_iter = 0, 0

        # Train
        for epoch in range(nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, nb_epochs))
            G_losses, D_losses = utils.AvgMeter(), utils.AvgMeter()
            start_epoch = datetime.datetime.now()

            avg_time_per_batch = utils.AvgMeter()
            # Mini-batch SGD
            for batch_idx, (x, _) in enumerate(data_loader):

                # Critic update ratio
                if self.gan_type == 'wgan':
                    n_critic = 20 if g_iter < 50 or (g_iter + 1) % 500 == 0 else self.n_critic
                else:
                    n_critic = self.n_critic

                # Training mode
                self.gan.G.train()

                # Discard last examples to simplify code
                if x.size(0) != self.batch_size:
                    break
                batch_start = datetime.datetime.now()

                # Print progress bar
                utils.progress_bar(batch_idx, self.batch_report_interval,
                    G_losses.avg, D_losses.avg)

                x = Variable(x)
                if torch.cuda.is_available() and self.use_cuda:
                    x = x.cuda()

                self.freq_weight = (epoch+1)/nb_epochs
                # Update discriminator
                D_loss, fake_imgs = self.gan.train_D(x, self.freq_weight, self.D_optimizer, self.batch_size)
                D_losses.update(D_loss, self.batch_size)
                d_iter += 1

                # Update generator
                if batch_idx % n_critic == 0:
                    G_loss = self.gan.train_G(self.freq_weight, self.G_optimizer, self.batch_size)
                    G_losses.update(G_loss, self.batch_size)
                    g_iter += 1

                batch_end = datetime.datetime.now()
                batch_time = int((batch_end - batch_start).total_seconds() * 1000)
                avg_time_per_batch.update(batch_time)

                # Report model statistics
                if (batch_idx % self.batch_report_interval == 0 and batch_idx) or \
                    self.batch_report_interval == self.num_batches:
                    G_all_losses.append(G_losses.avg)
                    D_all_losses.append(D_losses.avg)
                    utils.show_learning_stats(batch_idx, self.num_batches, G_losses.avg, D_losses.avg, avg_time_per_batch.avg)
                    [k.reset() for k in [G_losses, D_losses, avg_time_per_batch]]

                # Save stats
                if batch_idx % self.save_stats_interval == 0 and batch_idx:
                    stats = dict(G_loss=G_all_losses, D_loss=D_all_losses)
                    self.save_stats(stats)

            # Save model
            utils.clear_line()
            print('Elapsed time for epoch: {}'.format(utils.time_elapsed_since(start_epoch)))
            self.gan.save_model(self.ckpt_path, epoch, False)
            # Generating
            model.test(epoch)

        # Print elapsed time
        elapsed = utils.time_elapsed_since(start)
        print('Training done! Total elapsed time: {}\n'.format(elapsed))

        return G_loss, D_loss


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Generative adversarial network (GAN) implementation in PyTorch')
    parser.add_argument('-d', '--ckpt', help='checkpoint path', metavar='PATH',
        default='./checkpoint_ocatve')
    parser.add_argument('-t', '--type', help='model type',
        action='store', choices=['gan', 'wgan', 'lsgan'], default='gan', type=str)
    parser.add_argument('-o', '--optimizer', help='sgd optimizer',
        choices=['adam', 'rmsprop'], default='adam', type=str)
    parser.add_argument('-lr', '--learning-rate', help='learning rate',
        default=0.0002, type=float)
    parser.add_argument('-bs', '--batch-size', help='sgd minibatch size',
        default=64, type=int)
    parser.add_argument('-n', '--nb-epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-c', '--critic', help='d/g update ratio (critic)', default=1, type=int)
    parser.add_argument('-s', '--seed', help='random seed for debugging', type=int)
    parser.add_argument('-gpu', '--cuda', help='use cuda', action='store_true')
    args = parser.parse_args()

    # GAN parameters (type and latent dimension size)
    gan_params = {
        'gan_type': args.type,
        'latent_dim': 100,
        'n_critic': args.critic
    }

    # Training parameters (saving directory, learning rate, optimizer, etc.)
    train_params = {
        'root_dir': '/home/duralllopez/DATASETS/resized_celebA_128/',
        'batch_size': 64,
        'train_len': 202599,
        'learning_rate': args.learning_rate,
        'momentum': (0.5, 0.999),
        'optim': args.optimizer,
        'use_cuda': args.cuda
    }

    # Checkpoint parameters (report interval size, directories)
    ckpt_params = {
        'batch_report_interval': 100,
        'ckpt_path': args.ckpt,
        'save_stats_interval': 500
    }
    
    # Ready to train
    model = CelebA(train_params, ckpt_params, gan_params)

    # CelebA Dataset train_params['batch_size']
    data_loader = utils.load_dataset(train_params['root_dir'], train_params['batch_size'])

    if args.seed:
        torch.manual_seed(args.seed)

    # Train
    model.train(args.nb_epochs, data_loader)
    
    # Generating
    #for z in range (50):
        #model.test(z)

