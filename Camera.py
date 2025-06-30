import argparse
import os
import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from utils import resize_img, nms, find_max_box, crop_img, transform_pixel_v2
from Config import cfg
from Prompt import Face_prompt
import torchvision.transforms as transforms

def draw_landmark(landmark, image):
    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    return image

class Detector:

    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        assert os.path.exists(self.model_file)
        model = onnx.load(model_file)
        onnx.checker.check_model(model)
        self.session = onnxruntime.InferenceSession(self.model_file, None)

    def preprocess(self, img):
        pass

    def forward(self, img, score_thresh):
        pass

    def detect(self, img, score_thresh=0.5, mode='ORIGIN'):
        pass


class YUNET(Detector):

    def __init__(self, model_file=None, nms_thresh=0.5) -> None:
        super().__init__(model_file, nms_thresh)
        self.taskname = 'yunet'
        self.priors_cache = []
        self.strides = [8, 16, 32]
        self.NK = 5

    def forward(self, img, score_thresh):

        input_size = tuple(img.shape[0:2][::-1])
        blob = np.transpose(img, [2, 0, 1]).astype(np.float32)[np.newaxis,
                                                               ...].copy()
        nets_out = self.session.run(None,
                                    {self.session.get_inputs()[0].name: blob})
        scores, bboxes, kpss = [], [], []
        for idx, stride in enumerate(self.strides):
            cls_pred = nets_out[idx].reshape(-1, 1)
            obj_pred = nets_out[idx + len(self.strides)].reshape(-1, 1)
            reg_pred = nets_out[idx + len(self.strides) * 2].reshape(-1, 4)
            kps_pred = nets_out[idx + len(self.strides) * 3].reshape(
                -1, self.NK * 2)

            anchor_centers = np.stack(
                np.mgrid[:(input_size[1] // stride), :(input_size[0] //
                                                       stride)][::-1],
                axis=-1)
            anchor_centers = (anchor_centers * stride).astype(
                np.float32).reshape(-1, 2)

            bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
            bbox_wh = np.exp(reg_pred[:, 2:]) * stride
            tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
            tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
            br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
            br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

            bboxes.append(np.stack([tl_x, tl_y, br_x, br_y], -1))
            per_kps = np.concatenate(
                [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                 for i in range(self.NK)],
                axis=-1)

            kpss.append(per_kps)
            scores.append(cls_pred * obj_pred)

        scores = np.concatenate(scores, axis=0).reshape(-1)
        bboxes = np.concatenate(bboxes, axis=0)
        kpss = np.concatenate(kpss, axis=0)
        score_mask = (scores > score_thresh)
        scores = scores[score_mask]
        bboxes = bboxes[score_mask]
        kpss = kpss[score_mask]
        return (bboxes, scores, kpss)

    def detect(self, img, score_thresh=0.5, mode='ORIGIN'):
        det_img, det_scale = resize_img(img, mode)
        # det_img = cv2.resize(img, (640, 640))

        bboxes, scores, kpss = self.forward(det_img, score_thresh)

        bboxes /= det_scale
        kpss /= det_scale
        pre_det = np.hstack((bboxes, scores[:, None]))
        keep = nms(pre_det, self.nms_thresh)
        kpss = kpss[keep, :]
        bboxes = pre_det[keep, :]
        return bboxes, kpss

def parse_args():
    parser = argparse.ArgumentParser(description='Camera')

    parser.add_argument('--modelDir', type=str, default='./Weights/Face_prompt_base.pth')
    parser.add_argument('--detectDir', type=str, default='./Weights/yunet_n_640_640.onnx')
    parser.add_argument('--video_source', type=str, default='./Videos/angry.mp4')
    parser.add_argument('--points_number', type=float, default=29)
    parser.add_argument('--nms_thresh', type=float, default=0.45)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    model_file = args.detectDir
    detector = YUNET(model_file, nms_thresh=args.nms_thresh)
    score_thresh = args.nms_thresh

    model = Face_prompt(cfg.MODEL.TYPE)

    checkpoint_file = args.modelDir
    checkpoint = torch.load(checkpoint_file)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.module.load_state_dict(checkpoint)

    if args.points_number == 29:
        prompt_file = np.load('Prompt/shape_29.npz')['offset'] / 256.0
    elif args.points_number == 68:
        prompt_file = np.load('Prompt/shape_68.npz')['offset'] / 256.0
    elif args.points_number == 98:
        prompt_file = np.load('Prompt/shape_98.npz')['offset'] / 256.0
    elif args.points_number == 314:
        prompt_file = np.load('Prompt/shape_314.npz')['offset'] / 256.0
    else:
        raise NotImplementedError

    prompt_file = torch.from_numpy(prompt_file).float().cuda().unsqueeze(0)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    cap = cv2.VideoCapture(args.video_source)

    with torch.no_grad():
        while True:
            _, frame = cap.read()
            if frame is None: break

            bboxes, kpss = detector.detect(frame, score_thresh=score_thresh, mode='640,640')
            bbox = find_max_box(bboxes)

            input_image, trans = crop_img(frame.copy(), bbox, normalize)

            outputs_initial = model(input_image.cuda(), prompt_file)
            output = outputs_initial[0, -1, :, :].cpu().numpy()

            landmark = transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)
            frame = draw_landmark(landmark, frame)

            cv2.imshow('test', frame)
            cv2.waitKey(1)




