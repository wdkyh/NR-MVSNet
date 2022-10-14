import os, sys
import argparse
import glob
import errno
import os.path as osp
import shutil


parser = argparse.ArgumentParser()

parser.add_argument('--depth_dir', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--points_dir', type=str)

args = parser.parse_args()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def collect_dtu(args):
    mkdir_p(args.points_dir)
    all_scenes = sorted(glob.glob(args.depth_dir+'/*'))
    all_scenes = list(filter(os.path.isdir, all_scenes))
    for scene in all_scenes:
        if not scene.strip().split('/')[-1].startswith('scan'): continue
        scene_id = int(scene.strip().split('/')[-1][len('scan'):])
        all_plys = sorted(glob.glob('{}/points_mvsnet/consistencyCheck*'.format(scene)))
        print('Found points: ', all_plys)

        shutil.copyfile(all_plys[-1]+'/final3d_model.ply', '{}/mvsnet{:03d}_l3.ply'.format(args.points_dir, scene_id))

def collect_tanks(args):
    intermediate = [
        'Family', 'Francis', 'Horse', 'Lighthouse',
        'M60', 'Panther', 'Playground', 'Train'
    ]
    advanced = [
        'Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple'
    ]
    mkdir_p(args.points_dir)
    all_scenes = sorted(glob.glob(args.depth_dir + '/*'))
    for scene in all_scenes:
        if scene.strip().split('/')[-1].startswith('0_'): continue
        scene_name = scene.strip().split('/')[-1]
        if scene_name in intermediate:
            log_path = 'data/tanks_test/intermediate/{}/{}.log'.format(scene_name, scene_name)
        elif scene_name in advanced:
            log_path = 'data/tanks_test/advanced/{}/{}.log'.format(scene_name, scene_name)
        else:
            raise NotImplementedError
        shutil.copyfile(
            log_path, '{}/{}.log'.format(args.points_dir, scene_name)
        )

def collect_eth3d(args):
    mkdir_p(args.points_dir)
    all_scenes = sorted(glob.glob(args.depth_dir + '/*'))
    for scene in all_scenes:
        if scene.strip().split('/')[-1].startswith('0_'): continue
        scene_name = scene.strip().split('/')[-1]
        with open('{}/{}.txt'.format(args.points_dir, scene_name), 'w') as f:
            f.write('runtime 0.40')


if __name__ == '__main__':
    assert args.dataset in ['dtu', 'tanks', 'eth3d']
    if args.dataset == 'dtu': collect_dtu(args)
    if args.dataset == 'tanks': collect_tanks(args)
    if args.dataset == 'eth3d': collect_eth3d(args)
