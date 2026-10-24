"""Licence plate detection (YOLOv8) + text recognition (EasyOCR).

Usage (from the repository root):
    python src/predict_with_ocr.py model=weights/best.pt source=assets/sample_car.jpg

Any YOLOv8 predict option can be passed the same way (conf=0.5, show=True, ...).
Annotated outputs are saved under runs/detect/.
"""
import sys
from pathlib import Path

# Make the bundled `ultralytics` package (repo root) importable when running from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import cv2
import easyocr
import hydra
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

OCR_CONF = 0.2
reader = None  # EasyOCR reader, created lazily (slow to load)


def get_ocr(im, coors):
    """Crop the detected plate and read its text with EasyOCR."""
    global reader
    if reader is None:
        reader = easyocr.Reader(["en"])

    x1, y1, x2, y2 = (int(c) for c in coors[:4])
    crop = im[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)

    ocr = ""
    for _, text, conf in results:
        if len(results) == 1:
            ocr = text
        elif len(text) > 6 and conf > OCR_CONF:
            ocr = text
    return str(ocr)


class DetectionPredictor(BasePredictor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ocr_results = []

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % tuple(im.shape[2:])
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = get_ocr(im0, xyxy)
                if ocr:
                    self.ocr_results.append(ocr)
                    label = ocr  # draw the plate text on the image
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(xyxy,
                             im0.copy(),
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or str(ROOT_DIR / "weights" / "best.pt")
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()
    print("OCR Results:", predictor.ocr_results)
    return predictor.ocr_results


if __name__ == "__main__":
    predict()
