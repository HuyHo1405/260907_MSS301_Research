import re

# Bảng tra cứu ánh xạ ký tự hay nhầm lẫn giữa chữ và số trong OCR
CHAR_TO_NUM = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8', 'R': '8'
}

NUM_TO_CHAR = {
    '0': 'D',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'G'
}

VALID_PROVINCES = set([f"{i:02d}" for i in range(11, 100)])


def clean_alphanumeric(text):
    """Loại bỏ ký tự đặc biệt, giữ lại chữ hoa và số."""
    if not text:
        return ""
    return re.sub(r'[^A-Za-z0-9]', '', str(text)).upper()


def format_vietnamese_plate(texts):
    """
    Chuẩn hóa kết quả đọc thô OCR thành định dạng chuẩn biển số xe Việt Nam.
    Hỗ trợ cả biển 1 dòng và biển 2 dòng (biển vuông ô tô / xe máy).

    Quy chuẩn biển số Việt Nam:
      - Biển 5 số: [Tỉnh][Seri]-[Số].[Số] (VD: 51A-222.22, 68G1-668.86)
      - Biển 4 số: [Tỉnh][Seri]-[Số] (VD: 29X1-1234)
      - Biển 2 dòng: Dòng 1 (Seri tỉnh), Dòng 2 (Chuỗi số đăng ký)
    """
    if not texts:
        return ""

    # 1. Làm sạch ký tự
    clean_lines = [clean_alphanumeric(t) for t in texts if clean_alphanumeric(t)]
    if not clean_lines:
        return ""

    raw_combined = "".join(clean_lines)

    # 2. Khớp mẫu biển số 5 số (VD: 51A22222 -> 51A-222.22, 68G166886 -> 68G1-668.86)
    m5 = re.match(r'^([0-9]{2}[A-Z0-9]{1,3})([0-9]{3})([0-9]{2})$', raw_combined)
    if m5:
        seri, p1, p2 = m5.groups()
        return f"{seri}-{p1}.{p2}"

    # 3. Khớp mẫu biển số 4 số (VD: 29X11234 -> 29X1-1234)
    m4 = re.match(r'^([0-9]{2}[A-Z0-9]{1,3})([0-9]{4})$', raw_combined)
    if m4:
        seri, num = m4.groups()
        return f"{seri}-{num}"

    # 4. Trường hợp đọc được 2 dòng phân tách rõ rệt (dòng 1: seri, dòng 2: số)
    if len(clean_lines) >= 2:
        seri = clean_lines[0]
        num = clean_lines[1]
        num_clean = "".join([CHAR_TO_NUM.get(c, c) for c in num if CHAR_TO_NUM.get(c, c).isdigit()])
        if len(num_clean) == 5:
            num_fmt = f"{num_clean[:3]}.{num_clean[3:]}"
        elif len(num_clean) == 4:
            num_fmt = f"{num_clean[:2]}.{num_clean[2:]}"
        else:
            num_fmt = num
        return f"{seri}-{num_fmt}"

    return raw_combined
