#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:10:09 2019

@author: vijay
"""



import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from generator import Generator
import config as cf
import image_loader as il
import visualize as vz
from discriminator import Discriminator


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and cf.ngpu > 0) else "cpu")

# Create the generator
netG = Generator(cf.ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (cf.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(cf.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(il.weights_init)

# Print the model
print(netG)


# Create the Discriminator
netD = Discriminator(cf.ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (cf.ngpu > 1):
    netD = nn.DataParallel(netD, list(range(cf.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(il.weights_init)

# Print the model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, cf.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=cf.lr, betas=(cf.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=cf.lr, betas=(cf.beta1, 0.999))


# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []

    

def train():
    # Training Loop
    iters = 0
   
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(cf.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(il.dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, cf.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, cf.num_epochs, i, len(il.dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == cf.num_epochs-1) and (i == len(il.dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                        
                        iters += 1
                        
                        
                        
train()


vz.lossvstrain(G_losses, D_losses)
vz.generatorprogress(img_list)
vz.realvfake(device, img_list)