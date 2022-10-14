import argparse, os, sys, time, gc, datetime
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *


cudnn.benchmark = True
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='train', help='train or test', choices=['train', 'test'])
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--device', default='cuda', help='select model')

parser.add_argument('--dataset', default='blended', help='select dataset')
parser.add_argument('--testpath',default='data/blended_train/', help='test datapath')
parser.add_argument('--trainpath',default='data/blended_train/', help='train datapath')

parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')

parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--resume', action='store_true',help='continue to train the model')
parser.add_argument('--logdir', default='./results', help='the directory to save results/logs')

parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')

parser.add_argument('--num_range_depths', type=str, default="48,24,8", help="number of range sample of depth")
parser.add_argument('--num_normal_depths', type=str, default="16,8,0", help='number of normal sample of depth')
parser.add_argument('--iterations', type=str, default="1,1,1", help='number of each stage iteration')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--update_loss', type=float, default=3.0, help='update moudle loss weight')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')

parser.add_argument('--opt-level', type=str, default="O0")
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
parser.add_argument('--using_apex', action='store_true', help='using apex, need to install apex')


class NanError(Exception):
    pass


def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=lr_gamma, warmup_factor=1.0/3,
                                     warmup_iters=500, last_epoch=len(TrainImgLoader) * start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Epoch {}:'.format(epoch_idx + 1))

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, args)
            lr_scheduler.step()
            if (not is_distributed) or (dist.get_rank() == 0):
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                    print(
                       "Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss = {:.3f}, eval 4mm {:.2f}, time = {:.3f}".format(
                           epoch_idx + 1, args.epochs, batch_idx, len(TrainImgLoader),
                           optimizer.param_groups[0]["lr"], loss,
                           1 - scalar_outputs["thres4mm_error"],
                           time.time() - start_time))
                del scalar_outputs, image_outputs

        # checkpoint
        if (not is_distributed) or (dist.get_rank() == 0):
            if (epoch_idx + 1) % args.save_freq == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        if (epoch_idx + 1) % args.eval_freq == 0 or (epoch_idx + 1) == args.epochs:
            avg_test_scalars = DictAverageMeter()
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                global_step = len(TestImgLoader) * epoch_idx + batch_idx
                do_summary = global_step % args.summary_freq == 0
                loss, scalar_outputs, image_outputs = test_sample(model, model_loss, sample, args)
                if (not is_distributed) or (dist.get_rank() == 0):
                    if do_summary:
                        save_scalars(logger, 'test', scalar_outputs, global_step)
                        save_images(logger, 'test', image_outputs, global_step)
                        print("Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}".format(
                                                                            epoch_idx + 1, args.epochs,
                                                                            batch_idx,
                                                                            len(TestImgLoader), loss,
                                                                            time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs, image_outputs

            if (not is_distributed) or (dist.get_rank() == 0):
                save_scalars(logger, 'fulltest', avg_test_scalars.mean(), epoch_idx + 1)
                print("avg_test_scalars:", avg_test_scalars.mean())
            gc.collect()


def test(model, model_loss, TestImgLoader, args):
    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(model, model_loss, sample, args)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        if (not is_distributed) or (dist.get_rank() == 0):
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx, len(TestImgLoader), loss,
                                                                        time.time() - start_time))
            if batch_idx % 100 == 0:
                print("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader), avg_test_scalars.mean()))
    if (not is_distributed) or (dist.get_rank() == 0):
        print("final", avg_test_scalars.mean())


def train_sample(model, model_loss, optimizer, sample, args):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.num_range_depths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["proj_mats"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]
    
    try:
        loss = model_loss(
            outputs,
            depth_gt_ms,
            mask_ms,
            update_loss=args.update_loss,
            dlossw=[float(e) for e in args.dlossw.split(",") if e]
        )
        if np.isnan(loss.item()):
            print('NanError')
            raise NanError
    except:
        device, dtype = depth_gt.device, depth_gt.dtype
        loss = torch.tensor(0, dtype=dtype, device=device, requires_grad=True)
        
    if is_distributed and args.using_apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask, 8),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask, 20)}

    image_outputs = {"depth_est": depth_est * mask.to(torch.float32),
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "errormap": (depth_est - depth_gt).abs() * mask.to(torch.float32),
                     }

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


@make_nograd_func
def test_sample(model, model_loss, sample, args):
    if is_distributed:
        model_eval = model.module
    else:
        model_eval = model
    model_eval.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.num_range_depths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model_eval(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    depth_est = outputs["depth"]

    loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    scalar_outputs = {"loss": loss,
                      "abs_depth_error": AbsDepthError_metrics(depth_est, depth_gt, mask),
                      "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask, 2),
                      "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask, 4),
                      "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask, 8),
                      "thres20mm_error": Thres_metrics(depth_est, depth_gt, mask, 20),

                      "thres2mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask, [0, 2.0]),
                      "thres4mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask, [2.0, 4.0]),
                      "thres8mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask, [4.0, 8.0]),
                      "thres20mm_abserror": AbsDepthError_metrics(depth_est, depth_gt, mask, [14.0, 20.0]),
                    }

    image_outputs = {"depth_est": depth_est * mask.to(torch.float32),
                     "depth_est_nomask": depth_est,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "errormap": (depth_est - depth_gt).abs() * mask.to(torch.float32)}

    if is_distributed:
        scalar_outputs = reduce_scalar_outputs(scalar_outputs)

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':

    # parse arguments and check
    args = parser.parse_args()

    # using sync_bn by using nvidia-apex, need to install apex.
    if args.sync_bn:
        assert args.using_apex, "must set using apex and install nvidia-apex"
    if args.using_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

    if args.resume:
        assert args.mode == "train"
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    set_random_seed(args.seed)
    device = torch.device(args.device)

    if (not is_distributed) or (dist.get_rank() == 0):
        # create logger for mode "train" and "testall"
        if args.mode == "train":
            if not os.path.isdir(args.logdir):
                os.makedirs(args.logdir)
            current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            print("current time", current_time_str)
            print("creating new summary file")
            logger = SummaryWriter(args.logdir)
        print("argv:", sys.argv[1:])
        print_args(args)

    # model, optimizer
    model_loss = mvsnet_loss

    iterations = [int(x) for x in args.iterations.split(",") if x]
    num_range_samples = [int(x) for x in args.num_range_depths.split(",") if x]
    num_normal_samples = [int(x) for x in args.num_normal_depths.split(",") if x]
    model = MVSNet(num_range_samples=num_range_samples, num_normal_samples=num_normal_samples, iterations=iterations).to(device)
    
    if args.sync_bn:
        if (not is_distributed) or (dist.get_rank() == 0):
            print("using apex synced BN")
        import apex
        model = apex.parallel.convert_syncbn_model(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    if (not is_distributed) or (dist.get_rank() == 0):
        print("start at epoch {}".format(start_epoch))
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.using_apex:
        # Initialize Amp
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                          loss_scale=args.loss_scale
                                          )

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    else:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, "train", 3, crop_wh=(640, 512))
    test_dataset = MVSDataset(args.testpath, "val", 3, crop_wh=(640, 512))

    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=dist.get_world_size(),
                                                           rank=dist.get_rank())

        TrainImgLoader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler, num_workers=1,
                                    drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)
    else:
        TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True,
                                    pin_memory=args.pin_m)
        TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False,
                                   pin_memory=args.pin_m)


    if args.mode == "train":
        train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)
    elif args.mode == "test":
        test(model, model_loss, TestImgLoader, args)
    else:
        raise NotImplementedError