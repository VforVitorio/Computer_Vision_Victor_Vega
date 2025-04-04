# USAGE
# python test.py

# import the necessary packages
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from pyimagesearch import config, utils
from pyimagesearch.network import Decoder, Encoder

# set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - " "%(levelname)s - %(message)s"
)

# generate a random input tensor with the same shape as the input images
# (1: batch size, config.CHANNELS: number of channels,
# config.IMAGE_SIZE: height, config.IMAGE_SIZE: width)
dummy_input = torch.randn(1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)

# create an encoder instance with the specified channels,
# image size, and embedding dimensions
# then move it to the device (CPU or GPU) specified in the config
encoder = Encoder(
    channels=config.CHANNELS,
    image_size=config.IMAGE_SIZE,
    embedding_dim=config.EMBEDDING_DIM,
).to(config.DEVICE)

# pass the dummy input through the encoder and
# get the output (encoded representation)
enc_out = encoder(dummy_input.to(config.DEVICE))

# get the shape of the tensor before it was flattened in the encoder
shape_before_flattening = encoder.shape_before_flattening

# create a decoder instance with the specified embedding dimensions,
# shape before flattening, and channels
# then move it to the device (CPU or GPU) specified in the config
decoder = Decoder(config.EMBEDDING_DIM, shape_before_flattening, config.CHANNELS).to(
    config.DEVICE
)

# load the saved state dictionaries for the encoder and decoder
checkpoint = torch.load(config.MODEL_WEIGHTS_PATH)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

# set the models to evaluation mode
encoder.eval()
decoder.eval()

# define the transformation to be applied to the data
transform = transforms.Compose([transforms.Pad(padding=2), transforms.ToTensor()])

# load the test data
testset = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=config.BATCH_SIZE, shuffle=True
)

logging.info("Creating and Saving Reconstructed Images with Trained Autoencoder")
# call the 'display_random_images' function from the 'utils' module to display
# and save random reconstructed images from the test data
# after the autoencoder training
utils.display_random_images(
    test_loader,
    encoder,
    decoder,
    title_recon="Reconstructed After Training",
    title_real="Real Test Images After Training",
    file_recon=config.FILE_RECON_AFTER_TRAINING,
    file_real=config.FILE_REAL_AFTER_TRAINING,
)

logging.info("Creating and Saving the Latent Space Plot of Trained Autoencoder")
# call the 'plot_latent_space' function from the 'utils' module to create a 2D
# scatter plot of the latent space representations of the test data
utils.plot_latent_space(test_loader, encoder, show=False)

logging.info(
    "Finally, Creating and Plotting the Linearly Separated Image (Grid) on "
    "Embeddings of Trained Autoencoder"
)
# Call the 'plot_image_grid_on_embeddings' function from the 'utils' module
# to create a grid of images linearly interpolated
# between embedding pairs in the latent space
utils.plot_image_grid_on_embeddings(test_loader, encoder, decoder)
