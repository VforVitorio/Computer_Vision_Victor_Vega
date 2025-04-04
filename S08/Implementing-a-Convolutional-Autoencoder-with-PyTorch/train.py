# USAGE
# python train.py

# import the necessary packages
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from pyimagesearch import config, utils
from pyimagesearch.network import Decoder, Encoder

# define the transformation to be applied to the data
transform = transforms.Compose([transforms.Pad(padding=2), transforms.ToTensor()])

# load the FashionMNIST training data and create a dataloader
trainset = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=config.BATCH_SIZE, shuffle=True
)

# Load the FashionMNIST test data and create a dataloader
testset = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=config.BATCH_SIZE, shuffle=True
)

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
# Dummy input
dummy_input = torch.randn(1, config.CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
enc_out = encoder(dummy_input.to(config.DEVICE))


# get the shape of the tensor before it was flattened in the encoder
shape_before_flattening = encoder.shape_before_flattening
# create a decoder instance with the specified embedding dimensions,
# shape before flattening, and channels
# then move it to the device (CPU or GPU) specified in the config
decoder = Decoder(config.EMBEDDING_DIM, shape_before_flattening, config.CHANNELS).to(
    config.DEVICE
)

# instantiate loss, optimizer, and scheduler
criterion = nn.BCELoss()
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=config.LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=config.PATIENCE, verbose=True
)

# call the 'display_random_images' function from the 'utils' module to display
# and save random reconstructed images from the test data
# before the autoencoder training
utils.display_random_images(
    test_loader,
    encoder,
    decoder,
    title_recon="Reconstructed Before Training",
    title_real="Real Test Images",
    file_recon=config.FILE_RECON_BEFORE_TRAINING,
    file_real=config.FILE_REAL_BEFORE_TRAINING,
)

# initialize the best validation loss as infinity
best_val_loss = float("inf")

# start training by looping over the number of epochs
for epoch in range(config.EPOCHS):
    print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
    # set the encoder and decoder models to training mode
    encoder.train()
    decoder.train()

    # initialize running loss as 0
    running_loss = 0.0

    # loop over the batches of the training dataset
    for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # move the data to the device (GPU or CPU)
        data = data.to(config.DEVICE)
        # reset the gradients of the optimizer
        optimizer.zero_grad()

        # forward pass: encode the data and decode the encoded representation
        encoded = encoder(data)
        decoded = decoder(encoded)

        # compute the reconstruction loss between the decoded output and
        # the original data
        loss = criterion(decoded, data)

        # backward pass: compute the gradients
        loss.backward()
        # update the model weights
        optimizer.step()

        # accumulate the loss for the current batch
        running_loss += loss.item()

    # compute the average training loss for the epoch
    train_loss = running_loss / len(train_loader)

    # compute the validation loss
    val_loss = utils.validate(encoder, decoder, test_loader, criterion)

    # print training and validation loss for current epoch
    print(
        f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f}"
    )

    # save best model weights based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
            config.MODEL_WEIGHTS_PATH,
        )

    # adjust learning rate based on the validation loss
    scheduler.step(val_loss)

    # save validation output reconstruction for the current epoch
    utils.display_random_images(
        data_loader=test_loader,
        encoder=encoder,
        decoder=decoder,
        file_recon=os.path.join(
            config.training_progress_dir, f"epoch{epoch + 1}_test_recon.png"
        ),
        display_real=False,
    )

print("Training finished!")
