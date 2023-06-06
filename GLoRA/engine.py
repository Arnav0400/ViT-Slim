import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils import save, log
from avalanche.evaluation.metrics.accuracy import Accuracy

def train(args, model, train_dl, test_dl, opt, scheduler, epoch):
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        for i, batch in enumerate(train_dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 100 == 99:
            acc = test(model, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
            save(args, model)
            pbar.set_description(str(acc) + '|' + str(args.best_acc))
            log(args, acc, ep)

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)

    return acc.result()
