# [Stabilizing GANs with Octave Convolutions](https://ift6135h18.wordpress.com)

## Dependencies
Tested on Python 3.6.x.
* [PyTorch](http://pytorch.org/)
* [NumPy](http://www.numpy.org/)



## CelebA dataset
The full [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is available [here](https://drive.google.com/open?id=1p6WtrxprsjsiedQJkKVoiqvdrP1m9BuF). To resize the RGB images to 128 by 128 pixels, run `resize_celeba.py`.

## Training
To train a model, simply specify the model type (`gan`, `wgan` or `lsgan`) with the appropriate hyperparameters. In case these parameters are not specified, the program reverts back to default training parameters from the original papers.

```
./train.py --type wgan \
           --nb-epochs 50 \
           --batch-size 64 \
           --learning-rate 0.00005 \
           --optimizer rmsprop \
           --critic 5 \
           --ckpt ./../checkpoints/trained_wgan \
           --cuda
```



## References

### GAN
>Arjovsky et al. [Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1701.07875). In *Proceedings of the 34th International Conference on Machine Learning*, ICML 2017.

>Goodfellow et al. [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661). In *Advances in Neural Information
Processing Systems 27: Annual Conference on Neural Information Processing Systems
2014*, 2014.

>Mao et al. [Multi-class Generative Adversarial Networks with the L2 Loss Function](https://arxiv.org/abs/1511.06434). arXiv,  abs/1611.04076, 2016.

>Radford et al. [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434). arXiv, abs/1511.06434, 2015.

### Octave Convolution
>Chen et al. [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049). arXiv, abs/1511.06434, 2015.

