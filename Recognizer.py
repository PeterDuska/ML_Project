import glob
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
from joblib import dump, load
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

CLASS_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "add": 10,
    "sub": 11,
    "eq": 12,
    "div": 13,
    "mul": 14,
}

DISPLAY_LABELS = {10: "+", 11: "-", 12: "=", 13: "/", 14: "*"}

DEFAULT_MODEL_PATH = "recognizer_mlp.joblib"


def preprocess_to_mnist28(pil_img: Image.Image) -> Optional[np.ndarray]:
    img = pil_img.convert("L")
    inv = ImageOps.invert(img)
    arr = np.array(inv).astype(np.float32)

    if arr.max() < 5:
        return None

    arr = (arr > 30).astype(np.float32) * arr
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = arr[y0 : y1 + 1, x0 : x1 + 1]

    h, w = crop.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))

    pil = Image.fromarray(crop.astype(np.uint8), mode="L")
    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    small = np.array(pil).astype(np.float32)

    canvas = np.zeros((28, 28), dtype=np.float32)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = small

    ys, xs = np.nonzero(canvas)
    if len(xs) > 0:
        weights = canvas[ys, xs]
        cx = (xs * weights).sum() / (weights.sum() + 1e-6)
        cy = (ys * weights).sum() / (weights.sum() + 1e-6)
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        canvas = np.roll(canvas, shift=(shift_y, shift_x), axis=(0, 1))

    canvas = np.clip(canvas / 255.0, 0.0, 1.0)
    return canvas.reshape(-1).astype(np.float32)


def preprocess_to_mnist28_image(pil_img: Image.Image) -> Optional[Image.Image]:
    feat = preprocess_to_mnist28(pil_img)
    if feat is None:
        return None
    arr = (feat.reshape(28, 28) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _ink_mask(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    inv = 255 - arr
    return inv > 30


def _fill_small_gaps_1d(flags: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0:
        return flags
    filled = flags.copy()
    gap_start = None
    for i, val in enumerate(flags):
        if val:
            if gap_start is not None and i - gap_start <= max_gap:
                filled[gap_start:i] = True
            gap_start = None
        elif gap_start is None:
            gap_start = i
    return filled


def _find_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    start = None
    for i, val in enumerate(flags):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _pad_box(box: List[int], w: int, h: int, pad: int) -> List[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    ]


def _find_equation_boxes(mask: np.ndarray, min_area: int) -> List[List[int]]:
    row_has = mask.any(axis=1)
    row_has = _fill_small_gaps_1d(row_has, 6)
    boxes = []
    for y0, y1 in _find_runs(row_has):
        band = mask[y0:y1]
        if band.sum() < min_area:
            continue
        col_has = band.any(axis=0)
        col_has = _fill_small_gaps_1d(col_has, 2)
        if not col_has.any():
            continue
        xs = np.where(col_has)[0]
        x0, x1 = xs[0], xs[-1] + 1
        boxes.append([x0, y0, x1, y1])
    return boxes


def _split_box_by_valley(
    mask: np.ndarray, box: List[int], min_area: int
) -> List[List[int]]:
    x0, y0, x1, y1 = box
    w_box = x1 - x0
    h_box = y1 - y0
    if w_box < 28 or w_box < int(1.35 * h_box):
        return [box]

    cols = mask[y0:y1, x0:x1].sum(axis=0).astype(np.float32)
    max_col = cols.max()
    if max_col <= 0:
        return [box]

    if cols.size >= 3:
        cols = np.convolve(cols, np.ones(3, dtype=np.float32) / 3.0, mode="same")

    left_bound = int(0.25 * w_box)
    right_bound = int(0.75 * w_box)
    if right_bound - left_bound < 5:
        return [box]

    mid_rel = left_bound + int(np.argmin(cols[left_bound:right_bound]))
    min_val = cols[mid_rel]
    if min_val > 0.1 * max_col:
        return [box]

    if mid_rel < 5 or (w_box - mid_rel) < 5:
        return [box]

    left_area = cols[:mid_rel].sum()
    right_area = cols[mid_rel:].sum()
    if left_area < min_area or right_area < min_area:
        return [box]

    split_x = x0 + mid_rel
    return [[x0, y0, split_x, y1], [split_x, y0, x1, y1]]


def _segment_equation_symbols(
    mask: np.ndarray, eq_box: List[int], min_area: int
) -> List[List[int]]:
    x0, y0, x1, y1 = eq_box
    band = mask[y0:y1, x0:x1]
    col_has = band.any(axis=0)
    col_has = _fill_small_gaps_1d(col_has, 1)
    boxes = []
    for sx0, sx1 in _find_runs(col_has):
        seg = band[:, sx0:sx1]
        if not seg.any():
            continue
        ys, xs = np.nonzero(seg)
        bx0 = x0 + sx0
        bx1 = x0 + sx1
        by0 = y0 + ys.min()
        by1 = y0 + ys.max() + 1
        if (bx1 - bx0) * (by1 - by0) < min_area:
            continue
        boxes.append([bx0, by0, bx1, by1])

    split_boxes = []
    for box in boxes:
        split_boxes.extend(_split_box_by_valley(mask, box, min_area))
    split_boxes.sort(key=lambda b: b[0])
    return split_boxes


def _apply_symbol_padding(
    boxes: List[List[int]], w: int, h: int, pad: int
) -> List[List[int]]:
    if not boxes:
        return boxes
    boxes.sort(key=lambda b: b[0])
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        left_pad = pad
        right_pad = pad
        if i > 0:
            gap_left = x0 - boxes[i - 1][2]
            left_pad = min(pad, max(0, gap_left // 2))
        if i + 1 < len(boxes):
            gap_right = boxes[i + 1][0] - x1
            right_pad = min(pad, max(0, gap_right // 2))

        x0 = max(0, x0 - left_pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + right_pad)
        y1 = min(h, y1 + pad)
        boxes[i] = [x0, y0, x1, y1]
    return boxes


def _segment_equations(
    mask: np.ndarray, w: int, h: int
) -> List[Tuple[List[int], List[List[int]]]]:
    min_area = 30
    eq_boxes = _find_equation_boxes(mask, min_area)
    results = []
    for eq_box in eq_boxes:
        symbol_boxes = _segment_equation_symbols(mask, eq_box, min_area)
        symbol_boxes = _apply_symbol_padding(symbol_boxes, w, h, pad=6)
        eq_box = _pad_box(eq_box, w, h, pad=6)
        results.append((eq_box, symbol_boxes))
    return results


def _find_symbol_boxes(pil_img: Image.Image) -> List[List[int]]:
    mask = _ink_mask(pil_img)
    if not mask.any():
        return []
    h, w = mask.shape
    equations = _segment_equations(mask, w, h)
    boxes = []
    for _eq_box, sym_boxes in equations:
        boxes.extend(sym_boxes)
    return boxes


def segment_symbols(pil_img: Image.Image) -> List[Image.Image]:
    img = pil_img.convert("L")
    boxes = _find_symbol_boxes(pil_img)
    return [img.crop((x0, y0, x1, y1)) for x0, y0, x1, y1 in boxes]


def segment_symbols_with_boxes(
    pil_img: Image.Image,
) -> List[Tuple[Tuple[int, int, int, int], Image.Image]]:
    img = pil_img.convert("L")
    boxes = _find_symbol_boxes(pil_img)
    crops = [img.crop((x0, y0, x1, y1)) for x0, y0, x1, y1 in boxes]
    return list(zip([tuple(b) for b in boxes], crops))


def segment_equations_with_boxes(
    pil_img: Image.Image,
) -> List[
    Tuple[Tuple[int, int, int, int], List[Tuple[Tuple[int, int, int, int], Image.Image]]]
]:
    img = pil_img.convert("L")
    mask = _ink_mask(pil_img)
    if not mask.any():
        return []
    h, w = mask.shape
    equations = _segment_equations(mask, w, h)
    results = []
    for eq_box, sym_boxes in equations:
        crops = [img.crop((x0, y0, x1, y1)) for x0, y0, x1, y1 in sym_boxes]
        results.append((tuple(eq_box), list(zip([tuple(b) for b in sym_boxes], crops))))
    return results


def load_symbol_dataset(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for folder, cid in CLASS_MAP.items():
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fp in glob.glob(os.path.join(folder_path, "*.*")):
            try:
                img = Image.open(fp)
                feat = preprocess_to_mnist28(img)
                if feat is None:
                    continue
                X_list.append(feat)
                y_list.append(cid)
            except Exception:
                continue
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def load_mnist_digits() -> Tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)
    return X, y


def train_13class_model(symbol_root_dir: str) -> Tuple[MLPClassifier, float]:
    X_mnist, y_mnist = load_mnist_digits()  # digits 0..9
    X_sym, y_sym = load_symbol_dataset(symbol_root_dir)

    op_mask = y_sym >= 10
    X_ops, y_ops = X_sym[op_mask], y_sym[op_mask]

    X_all = np.concatenate([X_mnist, X_ops], axis=0)
    y_all = np.concatenate([y_mnist, y_ops], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=20,
        early_stopping=True,
        n_iter_no_change=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc


class Recognizer:
    def __init__(self, model: MLPClassifier):
        self.model = model

    @classmethod
    def from_symbol_root(cls, symbol_root_dir: str) -> Tuple["Recognizer", float]:
        model, acc = train_13class_model(symbol_root_dir)
        return cls(model), acc

    @classmethod
    def load(cls, model_path: str) -> "Recognizer":
        model = load(model_path)
        return cls(model)

    @classmethod
    def load_or_train(
        cls, symbol_root_dir: str, model_path: str
    ) -> Tuple["Recognizer", Optional[float], bool]:
        if os.path.isfile(model_path):
            return cls.load(model_path), None, True
        model, acc = train_13class_model(symbol_root_dir)
        dump(model, model_path)
        return cls(model), acc, False

    def save(self, model_path: str) -> None:
        dump(self.model, model_path)

    def predict(self, pil_img: Image.Image) -> Tuple[Optional[str], Optional[float]]:
        feat = preprocess_to_mnist28(pil_img)
        if feat is None:
            return None, None

        x = feat.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        label = DISPLAY_LABELS.get(pred, str(pred))
        return label, conf
