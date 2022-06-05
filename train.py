import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.cuda import amp
import torch.utils.data  as Data

from tqdm import tqdm

import network
import dataset
import config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_checkpoint(model,optimizer,filename):
    print("--checkpoint--")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint,filename)

def load_checkpoint(checkpoint_file,model,optimizer,lr):
    print("--Loading checkpoint--")
    checkpoint = torch.load(checkpoint_file,map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train(D,opt_d,G,opt_g,L1_loss,BCE,g_scaler,d_scaler,loader):
    loop = tqdm(loader)

    for index,(x,y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with amp.autocast():
            y_fake = G(x)
            D_real = D(x,y)
            D_real_loss = BCE(D_real,torch.ones_like(D_real))
            D_fake = D(x,y_fake.detach())
            D_fake_loss = BCE(D_fake,torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss)

        D.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_d)
        d_scaler.update()

        with amp.autocast():
            D_fake = D(x,y_fake)
            G_fake_loss = BCE(D_fake,torch.ones_like(D_fake))
            L1 = L1_loss(y_fake,y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_g.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_g)
        g_scaler.update()

        if index %10 == 0:
            loop.set_postfix(D_real = torch.sigmoid(D_real).mean().item(),D_fake = torch.sigmoid(D_fake).mean().item(),D_loss = D_loss.mean().item(),G_loss = G_loss.mean().item())

def main():
    D = network.Discriminator().to(DEVICE)
    G = network.Generator().to(DEVICE)
    opt_d = optim.RMSprop(D.parameters(),lr = config.LR)
    opt_g = optim.Adam(G.parameters(),lr = config.LR,betas=(0.5,0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_data = dataset.dataset(config.DATA_PATH + "/train")
    train_loader = Data.DataLoader(train_data,batch_size = config.BATCH_SIZE,shuffle = True,num_workers = config.NUM_WORKERS)

    val_data = dataset.dataset(config.DATA_PATH + "/val")
    val_loader = Data.DataLoader(val_data,batch_size = 1,shuffle = False)

    g_scaler = amp.GradScaler()
    d_scaler = amp.GradScaler()

    #load_checkpoint("./checkpoint/Gen.pth.tar", G, opt_g, config.LR)
    #load_checkpoint("./checkpoint/Disc.pth.tar", D, opt_d, config.LR)

    G.train()
    D.train()
    for epoch in range(config.EPOCHS):
        train(D,opt_d,G,opt_g,L1_LOSS,BCE,g_scaler,d_scaler,train_loader)
        if epoch % 9 == 0:
            save_checkpoint(G, opt_g,filename="./checkpoint/Gen.pth.tar")
            save_checkpoint(D, opt_d,filename="./checkpoint/Disc.pth.tar")
        if epoch % 3 == 0:
            x,y = next(iter(val_loader))
            x,y = x.to(DEVICE),y.to(DEVICE)
            print("--eval--")
            G.eval()

            with torch.no_grad():
                y_fake = G(x)
                y_fake = y_fake*0.5 + 0.5
                save_image(y_fake, f"./result/result_{epoch}.png")
                save_image(x*0.5+0.5, f"./result/input_{epoch}.png")

            G.train()

if __name__=="__main__":
    main()