# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
sys.path.insert(0, './kapao')
import yaml

import cv2
import torch
import torch.backends.cudnn as cudnn

from kapao.utils.torch_utils import select_device, time_sync
from kapao.utils.general import check_img_size, xyxy2xywh
from kapao.utils.datasets import LoadImages, LoadStreams
from kapao.models.experimental import attempt_load
from kapao.val import run_nms, post_process_batch
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


def detect(opt):
    # Initialize
    with open(opt.config_kapao) as f:
        data = yaml.safe_load(f)  # load data dict

    data['imgsz'] = opt.imgsz
    data['conf_thres'] = opt.conf_thres
    data['iou_thres'] = opt.iou_thres
    data['use_kp_dets'] = not opt.no_kp_dets
    data['conf_thres_kp'] = opt.conf_thres_kp
    data['iou_thres_kp'] = opt.iou_thres_kp
    data['conf_thres_kp_person'] = opt.conf_thres_kp_person
    data['overwrite_tol'] = opt.overwrite_tol
    data['scales'] = opt.scales
    data['flips'] = [None if f == -1 else f for f in opt.flips]
    data['count_fused'] = False

    video_path = opt.source

    cudnn.benchmark = True
    device = select_device(opt.device, batch_size=1)
    print('Using device: {}'.format(device))

    print("Initializing kapao model...")
    model = attempt_load(opt.kapao_model, map_location=device)  # load FP32 model
    half = opt.half & (device.type != 'cpu')
    if half:  # half precision only supported on CUDA
        model.half()
    stride = int(model.stride.max())  # model stride

    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size

    if video_path == "0":
        dataset = LoadStreams(video_path, img_size=imgsz, stride=stride, auto=True)
    else:
        dataset = LoadImages(video_path, img_size=imgsz, stride=stride, auto=True)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    if opt.output != '':
        fileType = os.path.splitext(opt.output)[-1]
        if fileType == ".avi":
            forcc = cv2.VideoWriter_fourcc(*'XVID')
        elif fileType == ".mp4":
            forcc = cv2.VideoWriter_fourcc(*'MP4V')
        else:
            raise ValueError("Please insert right type!")

        if video_path == "0":
            h, w, _ = dataset.imgs[0].shape
        else:
            h, w = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = cv2.VideoWriter(opt.output, forcc, 30.0, (w, h))

    print("Initialize deepsort...")
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    deepsort = DeepSort(
        opt.deep_sort_model,
        device,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    )

    dt = [0.] * 4
    for i, (path, img, im0, _) in enumerate(dataset):
        t1 = time_sync() * 1000
        if video_path == "0":
            im0 = im0[0]
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync() * 1000
        dt[0] = t2 - t1

        out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
        t3 = time_sync() * 1000
        dt[1] = t3 - t2
        person_dets, kp_dets = run_nms(data, out)
        bboxes, poses, score, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)
        t4 = time_sync() * 1000
        dt[2] = t4 - t3

        for i, det in enumerate(person_dets):
            if det is not None and len(det):
                det = det[:, :6]
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                out = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0, poses)      # [[x1, y1, x2, y2, id, class, prob]]
        outputs, poses = out

        t5 = time_sync() * 1000
        dt[3] = t5 - t4

        print("Speed: {:.2f}ms pre-process, {:.2f}ms inference, {:.2f}ms NMS, {:.2f}ms deep sort update".format(*dt))

        im0_copy = im0.copy()
        for j, (bbox, pose) in enumerate(zip(outputs, poses)):
            x1, y1, x2, y2 = bbox[0: 4]
            cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), opt.color, thickness=1)
            cv2.putText(im0_copy, "id: {}, prob: {:.2f}".format(int(bbox[4]), bbox[-1]), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            for x, y, c in pose[data['kp_face']]:
                cv2.circle(im0_copy, (int(x), int(y)), opt.kp_size, opt.color, opt.kp_thick)
            for seg in data['segments'].values():
                pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                cv2.line(im0_copy, pt1, pt2, opt.color, opt.line_thick)

        cv2.imshow('', im0_copy)
        if opt.output != '':
            writer.write(im0_copy)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()

    if opt.output != '':
        writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kapao_model', nargs='+', type=str, default='kapao/weights/kapao_s_coco.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', default='mobilenetv2_x1_0_market1501', type=str)
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', default='', type=str, help='output video')  # output video
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--imgsz', type=int, default=1024, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--overwrite-tol', type=int, default=50)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])

    parser.add_argument('--color', type=int, nargs='+', default=[255, 255, 0], help='pose color')
    parser.add_argument('--kp-size', type=int, default=2, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=2, help='keypoint circle thickness')
    parser.add_argument('--line-thick', type=int, default=3, help='line thickness')

    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--config_kapao", type=str, default="kapao/data/coco-kp.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_args()

    print(opt.output)
    with torch.no_grad():
        detect(opt)
