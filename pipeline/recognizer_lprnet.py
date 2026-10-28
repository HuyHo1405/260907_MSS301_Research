import os
import re
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from .postprocess import format_vietnamese_plate, CHAR_TO_NUM, NUM_TO_CHAR, VALID_PROVINCES

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O', '-'
]


class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SmallBasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            SmallBasicBlock(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = x.contiguous()
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = []
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


def ctc_greedy_decode(logits, chars=CHARS):
    arg_maxes = torch.argmax(logits, dim=1).cpu().numpy()
    decoded_results = []
    blank_idx = len(chars) - 1

    for b in range(arg_maxes.shape[0]):
        seq = arg_maxes[b]
        char_list = []
        prev_idx = -1
        for idx in seq:
            if idx != prev_idx and idx != blank_idx:
                char_list.append(chars[idx])
            prev_idx = idx
        decoded_results.append("".join(char_list))
    return decoded_results


def split_plate_lines(roi):
    h, w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_sum = np.sum(binary, axis=1)

    min_idx = int(h * 0.45)
    min_val = float('inf')
    for y in range(int(h * 0.35), int(h * 0.65)):
        if row_sum[y] < min_val:
            min_val = row_sum[y]
            min_idx = y

    top_crop = roi[0:min_idx, 0:w]
    bot_crop = roi[min_idx:h, 0:w]
    return top_crop, bot_crop


class LPRNetRecognizer:
    """
    OCR Recognizer Engine dựa trên PyTorch LPRNet (Lightweight CNN + CTC Decoder).
    Thích hợp cho nghiên cứu so sánh cấu trúc mạng nơ-ron chuyên dụng.
    """
    def __init__(self, weights_path='weights/lprnet.pth', img_size=(94, 24), device=None):
        self.img_size = img_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        resolved_path = weights_path
        if not os.path.exists(resolved_path):
            alt_path = os.path.basename(weights_path)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            elif os.path.exists('weights/lprnet.pth'):
                resolved_path = 'weights/lprnet.pth'
            elif os.path.exists('lprnet.pth'):
                resolved_path = 'lprnet.pth'

        self.net = LPRNet(class_num=len(CHARS)).to(self.device)
        self.net.eval()

        if os.path.exists(resolved_path):
            state_dict = torch.load(resolved_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"✅ Đã nạp LPRNet weights: '{resolved_path}'")
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file weights LPRNet '{weights_path}'")

    def _preprocess_crop(self, crop):
        img_resized = cv2.resize(crop, self.img_size)
        img_float = img_resized.astype('float32') - 127.5
        img_float *= 0.0078125
        tensor = torch.from_numpy(img_float.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
        return tensor

    def recognize(self, roi):
        """
        Nhận diện ký tự biển số từ vùng ảnh ROI.
        Input: roi (numpy.ndarray BGR)
        Output:
          - raw_texts: Danh sách các chuỗi text OCR thô đọc được
          - formatted_plate: Biển số chuẩn hóa theo quy chuẩn Việt Nam
          - duration_ms: Thời gian suy luận OCR (ms)
        """
        if roi is None or roi.size == 0:
            return [], "", 0.0

        t0 = time.perf_counter()
        h, w = roi.shape[:2]
        aspect_ratio = w / float(h)

        raw_texts = []
        with torch.no_grad():
            if aspect_ratio < 2.0:
                # Biển số 2 dòng (vuông) -> cắt 2 nửa
                top_part, bot_part = split_plate_lines(roi)
                t_top = self._preprocess_crop(top_part)
                t_bot = self._preprocess_crop(bot_part)
                batch = torch.cat([t_top, t_bot], dim=0)
                logits = self.net(batch)
                decoded = ctc_greedy_decode(logits)
                raw_texts = [decoded[0], decoded[1]]
            else:
                # Biển số 1 dòng (dài)
                t = self._preprocess_crop(roi)
                logits = self.net(t)
                decoded = ctc_greedy_decode(logits)
                raw_texts = [decoded[0]]

        duration_ms = (time.perf_counter() - t0) * 1000
        formatted_plate = format_vietnamese_plate(raw_texts)
        return raw_texts, formatted_plate, duration_ms
