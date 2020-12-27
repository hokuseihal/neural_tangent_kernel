import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import core as G

def operate(sec):
    if sec == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader

    with torch.set_grad_enabled(sec == 'train'):
        for idx, (data, target) in enumerate(loader):
            B,C,H,W=data.shape
            data = data.to(device)
            target = target.to(device)
            output = model(data.reshape(B,-1))
            loss = criterion(output, target)
            # acc=(abs(target-output)<0.05).float().mean()
            acc=(output.argmax(-1)==target).float().mean()
            if sec=='train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f'{e}, {idx}, {loss.item():.4f},{acc:.4f},{sec}')
            G.addvalue(writer,f'loss:{sec}',loss.item(),e)
            G.addvalue(writer,f'acc:{sec}',acc.item(),e)


if __name__ == '__main__':
    from net import NTK

    device = 'cpu'
    model = NTK(3*32 * 32, 10**4).to(device)
    optimizer=torch.optim.Adam(model.parameters())
    criterion=nn.CrossEntropyLoss()
    epoch=100
    bachsize=2048
    savefolder='out/exp_10000'
    writer={}
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data',
                                                                         train=True,
                                                                         download=True,
                                                                         transform=transforms.Compose(
                                                                             [transforms.ToTensor(),
                                                                              ])),
                                              batch_size=bachsize,
                                              shuffle=True,
                                              num_workers=8)
    valloader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data',
                                                                   train=False,
                                                                   download=True,
                                                                   transform=transforms.Compose(
                                                                       [transforms.ToTensor(),
                                                                        ])),
                                        batch_size=bachsize,
                                        shuffle=True,
                                        num_workers=8)

    for e in range(epoch):
        operate('train')
        operate('val')
        G.savedic(writer,savefolder,'')