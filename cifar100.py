from dataset import Cifar100Dataset
from model import *
from metrics.metrics import accuracy
from augmentations.augmentations import TrainAugment, TestAugment


# TODO:
# stochastic depth (Huang et al., 2016) with drop connect ratio 0.3.


def main():

    torch.manual_seed(2019)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2019)

    device = 'cuda:0'
    ###################################
    ###       Dataset
    ###################################
    train_dataset = Cifar100Dataset(root='./input/', train=True, download=True, transform=TrainAugment())
    test_dataset = Cifar100Dataset(root='./input/', train=False, download=True, transform=TestAugment())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                                   shuffle=True, num_workers=8,
                                                   pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                                  shuffle=False, num_workers=8,
                                                  pin_memory=False)

    #########################################
    # model
    model = efficientnet_b0(num_classes=100).to(device)
    # Optimizer
    # torch.optim : https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.016, momentum=0.9, weight_decay=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    # Scheduler
    # torch.optim.lr_scheduler : https://pytorch.org/docs/stable/optim.html?highlight=lr_scheduler#torch.optim.lr_scheduler.LambdaLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(2.4*len(train_dataloader)), gamma=0.97)
    # Loss and Evalaiton function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    eval_fn = accuracy

    epoch = 500

    for i in range(epoch):
        train_loss, train_accuracy = train(model, optimizer, train_dataloader,
                                               device, loss_fn, eval_fn, i, scheduler)
        test_loss, test_accuracy = test(model, test_dataloader,
                                            device, loss_fn, eval_fn)
        print(f'''========   epoch {i:>3} (lr: {scheduler.get_lr()[0]:.5f})  ========
train loss = {train_loss:.5f} | train err = {1-train_accuracy:.2%} |
test loss = {test_loss:.5f} | test err = {1-test_accuracy:.2%}''')

    print('=== Success ===')


def l2_loss(model):
    loss = 0.0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            for p in m.parameters():
                loss += (p ** 2).sum() / 2 #p.norm(2)

    return loss

def train(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        preds = logits.softmax(dim=1)
        loss = loss_fn(logits, targets.argmax(dim=1)) 
        loss += 1e-5 * l2_loss(model)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        avg_accuracy += eval_fn(preds, targets)
        if scheduler is not None:
            scheduler.step()
    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy


def test(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    avg_loss = 0
    avg_accuracy = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.softmax(dim=1)
            loss = loss_fn(logits, targets.argmax(dim=1)) 
            loss += 1e-5 * l2_loss(model)
            avg_loss += loss.item()
            avg_accuracy += eval_fn(preds, targets)

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy


if __name__ == '__main__':
    main()
