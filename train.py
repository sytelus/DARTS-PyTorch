import  os
import  sys
import  time
import  glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
from  genotypes import Genotype
import  torch.utils
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn
import pathlib
from timebudget import timebudget

from    model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar10")
parser.add_argument('--data', type=str, default='~/dataroot', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=96, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10000, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='~/logdir', help='experiment name')
parser.add_argument('--seed', type=float, default=0.0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.data = os.path.expanduser(args.data)
os.makedirs(args.data, exist_ok=True)

pt_output_dir = os.environ.get('PT_OUTPUT_DIR', '')
if pt_output_dir:
    args.exp_path = pt_output_dir
geno_path = os.path.join(os.path.expanduser(args.exp_path), 'darts_pytorch_orig_search', 'genotype.txt')
args.exp_path = os.path.join(os.path.expanduser(args.exp_path), 'darts_pytorch_orig_eval')
args.exp_path = utils.create_exp_dir(args.exp_path, scripts_to_save=glob.glob('*.py'))

args.save = args.exp_path
args.seed = int(args.seed)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():


    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    if not args.arch:
        geno_s = pathlib.Path(geno_path).read_text()
    else:
        geno_s = "genotypes.%s" % args.arch
    genotype = eval(geno_s)
    model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype).cuda()

    logging.info(f"seed = {args.seed}")
    logging.info(f"geno_s = {geno_s}")
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=0 if 'pydevd' in sys.modules else 2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=0 if 'pydevd' in sys.modules else 2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    lines = [f'epoch\ttrain_acc\tval_acc']
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc: %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc: %f', valid_acc)

        lines.append(f'{epoch}\t{train_acc}\t{valid_acc}')

    timebudget.report()
    utils.save(model, os.path.join(args.save, 'trained.pt'))
    print('saved to: trained.pt')
    pathlib.Path(os.path.join(args.exp_path, 'eval.tsv')).write_text('\n'.join(lines))

@timebudget
def train(train_queue, model, criterion, optimizer):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (x, target) in enumerate(train_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if (step+1) % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

@timebudget
def infer(valid_queue, model, criterion):

    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (x, target) in enumerate(valid_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits, _ = model(x)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        if (step+1) % args.report_freq == 0:
            logging.info('>>Validation: %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
