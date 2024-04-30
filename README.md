# Controllable Generation

## Overview
This notebook implements a GAN controllability method using gradients from a classifier. By training a classifier to recognize a relevant feature, the notebook demonstrates how to adjust the generator's inputs to control the generation process. The implementation is based on a pre-trained generator and classifier, allowing users to focus on the controllability aspects.

## Dataset
The notebook uses the CelebA dataset, which consists of annotated celebrity images. Since CelebA images are colored (RGB), the generator produces images with three channels for red, green, and blue.

## Packages and Visualization
- `torch`: PyTorch library for deep learning
- `torchvision`: PyTorch package containing popular datasets, model architectures, and image transformations
- `tqdm`: Progress bar library for tracking iterations
- `matplotlib.pyplot`: Library for plotting visualizations

## Generator and Noise
The notebook defines a `Generator` class responsible for generating images in the GAN model. It also provides a function for creating noise vectors (`get_noise`) to feed into the generator.

## Classifier
The `Classifier` class is used for classifying images as real or fake in the GAN model. It is trained on the CelebA dataset to distinguish between real and generated images.

## Specifying Parameters
Before training, users need to specify parameters such as noise vector dimension (`z_dim`), batch size (`batch_size`), and device type (`device`).

## Training
The notebook provides code for training the controllable generation model using gradient ascent to adjust noise vectors. It explains the process of updating noise vectors to produce more of a desired feature.

## Entanglement and Regularization
To address the challenges of entangled features, regularization techniques are discussed. The notebook includes a function (`get_score`) for penalizing changes to non-target features.

## Usage
Users can clone the repository and run the notebook in their preferred environment. Pretrained models for the generator and classifier are provided for convenience.

## Credits
The implementation is based on concepts from generative adversarial networks (GANs) and deep learning. Pretrained models are sourced from open datasets and libraries.

## License
This project is licensed under the [MIT License](LICENSE).
