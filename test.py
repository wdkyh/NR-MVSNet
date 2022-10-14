import argparse, os, time, gc, cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

from utils import *
from models import *
from datasets import find_dataset_def
from datasets.utils import save_pfm

cudnn.benchmark = True


parser = argparse.ArgumentParser(description='Predict depth')
parser.add_argument('--mode', default='test', help='select mode')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_test', help='select dataset')
parser.add_argument('--testpath',default='data/dtu_test', help='testing data dir for some scenes')

parser.add_argument('--loadckpt', default='checkpoints/DDRNet.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='results', help='output dir')

parser.add_argument('--num_range_depths', type=str, default="48,24,8", help="number of range sample of depth")
parser.add_argument('--num_normal_depths', type=str, default="16,8,0", help='number of normal sample of depth')
parser.add_argument('--iterations', type=str, default="1,1,1", help='number of each stage iteration')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

parser.add_argument('--num_view', type=int, default=5, help='num of view')
parser.add_argument('--height', type=int, default=864, help='testing max h')
parser.add_argument('--width', type=int, default=1152, help='testing max w')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')


# parse arguments and check
args = parser.parse_args()
num_stage = len([int(nd) for nd in args.num_range_depths.split(",") if nd])

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32)
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img


# run mvsnet model to save depth maps and confidence maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        args.testpath, args.mode, args.num_view, img_wh=(args.width, args.height)
    )
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    iterations = [int(x) for x in args.iterations.split(",") if x]
    num_range_samples = [int(x) for x in args.num_range_depths.split(",") if x]
    num_normal_samples = [int(x) for x in args.num_normal_depths.split(",") if x]
    model = MVSNet(
        num_range_samples=num_range_samples,
        num_normal_samples=num_normal_samples,
        iterations=iterations
    )

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)

            start_time = time.time()
            outputs = model(
                sample_cuda["imgs"], sample_cuda["proj_mats"], sample_cuda["depth_values"]
            )
            end_time = time.time()

            depth=outputs["depth"]
            confidence=outputs["photometric_confidence"]
            depth=tensor2numpy(depth)
            confidence=tensor2numpy(confidence)
            del sample_cuda
            imgs = sample["imgs"].numpy()
            filenames = sample["filename"]
            cams = sample["proj_mats"]["stage{}".format(num_stage)].numpy()

            # save depth maps and confidence maps
            for values in zip(filenames, cams, depth, confidence ):
                filename, cam, depth_est, confidence = values
                cam = cam[0]  #ref cam
                scan = filename.split('/')[0]
                height, width = depth_est.shape[-2:]
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)

                write_cam(cam_filename, cam)
                save_pfm(depth_filename, depth_est)
                save_pfm(confidence_filename, confidence)

                assert args.dataset in ['dtu_test', 'eth3d_test', 'tanks_test']
                if args.dataset in ["dtu_test" , "eth3d_test"]:
                    img = read_img(
                        os.path.join(args.testpath, filename.format('images', '.jpg')),
                        (width, height)
                    )
                else:
                    img = read_img(
                    os.path.join(args.testpath, args.mode, filename.format('images', '.jpg')),
                    (width, height)
                )  
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

                print('{}, Iter {}/{}, Time:{} Res:{}'.format(scan, batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    save_depth()