import argparse

from Config import cfg

from Prompt import Face_prompt
from Dataloader import All_Dataset

import torch
import numpy as np

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--dataset', type=str, default='300W')
    parser.add_argument('--modelDir', type=str, default='./Weights/Face_prompt_base.pth')
    parser.add_argument('--nms_thresh', type=float, default=0.45)

    args = parser.parse_args()

    return args


def calculate_loss(input_tensor, ground_truth):
    L2_Loss = np.linalg.norm((input_tensor - ground_truth), axis=1)
    if len(ground_truth) == 98:
        L2_norm = np.linalg.norm(ground_truth[60, :] - ground_truth[72, :], axis=0)
    elif len(ground_truth) == 68:
        L2_norm = np.linalg.norm(ground_truth[36, :] - ground_truth[45, :], axis=0)
    elif len(ground_truth) == 29:
        L2_norm = np.linalg.norm(ground_truth[8, :] - ground_truth[9, :], axis=0)
    else:
        raise NotImplementedError
    L2_Loss = np.mean(L2_Loss / L2_norm)
    return L2_Loss


def transform_pixel_v2(pt, trans, inverse=False):
    if inverse is False:
        pt = pt @ (trans[:,0:2].T) + trans[:,2]
    else:
        pt = (pt - trans[:,2]) @ np.linalg.inv(trans[:,0:2].T)
    return pt


def main_function():
    args = parse_args()

    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = Face_prompt(cfg.MODEL.TYPE)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    if args.dataset == 'WFLW':
        prompt_file = np.load('Prompt/shape_98.npz')['offset'] / 256.0
    elif args.dataset == '300W':
        prompt_file = np.load('Prompt/shape_68.npz')['offset'] / 256.0
    elif args.dataset == 'COFW':
        prompt_file = np.load('Prompt/shape_29.npz')['offset'] / 256.0
    else:
        raise NotImplementedError

    prompt_file = torch.from_numpy(prompt_file).float().cuda().unsqueeze(0)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = All_Dataset(
        cfg, cfg.ALL.ROOT, False, args.dataset,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.PIN_MEMORY
    )

    checkpoint_file = args.modelDir
    checkpoint = torch.load(checkpoint_file)
    model.module.load_state_dict(checkpoint)

    loss_list = []

    model.eval()

    with torch.no_grad():
        for i, (input, meta) in enumerate(valid_loader):
            outputs = model(input.cuda(), prompt_file)

            ground_truth = meta['initial'].cpu().numpy()[0]
            trans = meta['trans'].cpu().numpy()[0]

            output_stage = outputs[0, -1, :, :].cpu().numpy() * 256.0
            output_stage_trans = transform_pixel_v2(output_stage, trans, inverse=True)

            loss = calculate_loss(output_stage_trans, ground_truth)

            loss_list.append(loss)

            print("[{}]/[{}], loss: {}, loss_avg: {}".format(i+1, len(valid_loader), loss, np.mean(loss_list)))

        print("average loss on " + args.dataset +  " is {}%".format(np.mean(loss_list) * 100))

if __name__ == '__main__':
    main_function()

