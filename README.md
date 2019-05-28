# [Stabilizing GANs with Octave Convolutions](https://ift6135h18.wordpress.com)

## Dependencies
Tested on Python 3.6.x.
* [PyTorch](http://pytorch.org/) (1.0.1)
* [NumPy](http://www.numpy.org/) (1.16.2)



## CelebA dataset
The full [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is available [here](https://drive.google.com/open?id=1p6WtrxprsjsiedQJkKVoiqvdrP1m9BuF). To resize the RGB images to 128 by 128 pixels, run `resize_celeba.py`.

## Training
To train a model, simply specify the model type (`gan`, `wgan` or `lsgan`) with the appropriate hyperparameters. In case these parameters are not specified, the program reverts back to default training parameters from the original papers.

```
python train.py --type wgan \
           --nb-epochs 50 \
           --learning-rate 0.00005 \
           --optimizer rmsprop \
           --critic 5 \
           --cuda
```



## References

### GAN
>Arjovsky et al. [Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1701.07875).

>Goodfellow et al. [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).

>Mao et al. [Multi-class Generative Adversarial Networks with the L2 Loss Function](https://arxiv.org/abs/1511.06434).

>Radford et al. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). 

### Octave Convolution
>Chen et al. [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049).

