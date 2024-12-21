import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 

import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from torchvision.utils import save_image

from torchsummary import summary

import matplotlib.pyplot as plt

def get_fashinMNIST_dataloader(batch_size: int): 
    
    preprocess = transforms.Compose([
        transforms.ToTensor(), # convert images(PILImage) to torch tensors  
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    # FashionMNIST : https://github.com/zalandoresearch/fashion-mnist 
    fashion_mnist_data = datasets.FashionMNIST(
        root='MNIST_data/',
        train=True, 
        transform=preprocess, 
        download=True
    )

    data_loader = DataLoader(
        fashion_mnist_data,
        batch_size=batch_size,
        shuffle=True
    )

    return data_loader


class GeneratorModel(nn.Module): 
    def __init__(self): 
        super().__init__()

        # input (batch, 100) -> output (batch, 7*7*256)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=100, out_features=256*7*7),
            nn.BatchNorm1d(num_features=256*7*7), 
            nn.ReLU(),
        )
        
        # input (batch, *) -> output (batch, 256, 7, 7)
        self.reshape = nn.Unflatten(dim=1, unflattened_size=(256, 7, 7))

        # input (batch, 256, 7, 7) -> output (batch, 128, 7, 7)
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU()
        )

        # input (batch, 128, 7, 7) -> output (batch, 64, 14, 14)
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(),
        )
    
        # input (batch, 64, 14, 14) -> output (batch, 1, 28, 28)
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x): 
        x = self.fc1(x)
        x = self.reshape(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        return x
    

class DiscrimiatorModel(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        # input(batch, 1, 28, 28) -> (batch, 64, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=64), 
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.3)
        )

        # input(batch, 64, 14, 14) -> (batch, 128, 7, 7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128), 
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.3)
        )

        # input(batch, 128, 7, 7) -> (batch, *)
        self.flatten = nn.Flatten()

        # input(batch, *) -> (batch, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=128*7*7, out_features=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x) 
        return x 


class Generator:
    def __init__(self, model: GeneratorModel):
        self.model = model
        self.loss = nn.BCELoss()
        self.optim = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def run(self, x):
        return self.model(x)
    
    def update_param(self, disc_output)->float:
        self.optim.zero_grad()
        
        ones = torch.ones_like(disc_output).to(disc_output.device)
        loss = self.loss(disc_output, ones)
        loss.backward()
        
        self.optim.step()
        return loss.item() 


class Discriminator:
    def __init__(self, model: DiscrimiatorModel):
        self.model = model
        self.loss = nn.BCELoss()
        self.optim = optim.Adam(model.parameters(), lr = 0.0002) 

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def run(self, x):
        return self.model(x)
    
    def update_param(self, real_outputs, fake_outputs)->float:
        self.optim.zero_grad() 

        real_loss = self.loss(torch.ones_like(real_outputs),real_outputs)
        fake_loss = self.loss(torch.zeros_like(fake_outputs), fake_outputs)
        loss = real_loss + fake_loss     
        loss.backward() 

        self.optim.step()
        return loss.item()


def train_step(train_images, gen:Generator, disc:Discriminator, device): 
    batch = train_images.shape[0]
    # print(batch)
    # noise = torch.rand((batch, 100), requires_grad=True).to(device)
    noise = torch.normal(mean=0.0, std=1.0, size=(batch, 100)).to(device)

    # gen.train() 
    # disc.train()

    fake_images = gen.run(noise)
    # breakpoint()
    fake_outputs = disc.run(fake_images)
    # real_outputs = disc.run(train_images)

    # generator, discirminator  loss 
    gen_loss = gen.update_param(fake_outputs)
    # disc_loss = disc.update_param(real_outputs, fake_outputs) 

    return gen_loss, 0. #, disc_loss 


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'execute on : {device}')

    generator_model = GeneratorModel().to(device)
    print(f'generator summary >> ')
    summary(generator_model, (100, ))
    print() 

    discriminator_model = DiscrimiatorModel().to(device)
    print(f'discriminator summary >> ')
    summary(discriminator_model, (1, 28, 28))
    print() 

    gen = Generator(generator_model)
    disc = Discriminator(discriminator_model)

    gen.train() 
    disc.train()

    epoch = 5
    batch_size = 32 
    data_loader = get_fashinMNIST_dataloader(batch_size)
    
    gen_loss_trace = [] 
    disc_loss_trace = [] 
    for i in range(epoch):
        gen_loss = 0 
        disc_loss = 0

        for datas, _ in data_loader:
            datas = datas.to(device)
            gloss, dloss = train_step(datas, gen, disc, device)
            gen_loss += gloss
            disc_loss += dloss

        print(gen_loss/len(data_loader))
        gen_loss_trace.append(gen_loss/len(data_loader))    
        disc_loss_trace.append(disc_loss/len(data_loader))

    # print(gen_loss_trace)
    # batch = 32 
    # noise = torch.rand(batch, 100) 
    # generated_img = generator(noise)
    # output = discriminator(generated_img)


if __name__ == "__main__":
    main()

'''
for batch_idx, (images, labels) in enumerate(data_loader):
    print(f"Batch {batch_idx+1}:")
    print(f" - Images shape: {images.shape}")
    print(f" - Labels shape: {labels.shape}")
    break  # Remove this line to go through all batches

breakpoint()
''' 