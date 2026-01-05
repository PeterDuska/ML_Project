import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from Recognizer import DEFAULT_MODEL_PATH, Recognizer, segment_equations_with_boxes


CLASSES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "add",
    "sub",
    "mul",
    "div",
    "eq",
]

TOKEN_TO_CLASS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "=": "eq",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}

OPS = ["+", "-", "*", "/"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic a op b = equation images and evaluate the model."
    )
    parser.add_argument("--out-dir", default="synthetic_equations")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--max-eq", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--spacing", type=int, default=12)
    parser.add_argument("--row-gap", type=int, default=50)
    parser.add_argument("--pad-x", type=int, default=30)
    parser.add_argument("--pad-y", type=int, default=30)
    parser.add_argument("--max-result", type=int, default=99)
    parser.add_argument("--max-width-factor", type=float, default=1.3)
    parser.add_argument("--operator-height", type=float, default=0.85)
    parser.add_argument("--ink-threshold", type=int, default=210)
    parser.add_argument(
        "--no-strip-bottom",
        dest="strip_bottom",
        action="store_false",
        help="Do not remove thin bottom lines from symbols.",
    )
    parser.set_defaults(strip_bottom=True)
    return parser.parse_args()


def load_class_paths(root: Path) -> Dict[str, List[Path]]:
    paths: Dict[str, List[Path]] = {name: [] for name in CLASSES}
    for name in CLASSES:
        folder = root / name
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                paths[name].append(p)
    missing = [name for name, items in paths.items() if not items]
    if missing:
        raise RuntimeError(f"Missing class images for: {', '.join(missing)}")
    return paths


def strip_bottom_line(
    img: Image.Image, mask: Image.Image, min_ratio: float = 0.6, max_rows: int = 2
) -> Tuple[Image.Image, Image.Image]:
    w, h = mask.size
    if h == 0 or w == 0:
        return img, mask
    pixels = mask.load()
    rows_to_strip = 0
    for y in range(h - 1, max(-1, h - max_rows - 1), -1):
        ink = 0
        for x in range(w):
            if pixels[x, y] > 0:
                ink += 1
        if ink / w >= min_ratio:
            rows_to_strip += 1
        else:
            break
    if rows_to_strip:
        img = img.crop((0, 0, w, h - rows_to_strip))
        mask = mask.crop((0, 0, w, h - rows_to_strip))
    return img, mask


def is_operator_class(cls: str) -> bool:
    return cls in {"add", "sub", "mul", "div", "eq"}


def combine_tokens(tokens: List[str]) -> List[str]:
    expr_tokens: List[str] = []
    number_buf = ""
    for tok in tokens:
        if tok.isdigit():
            number_buf += tok
            continue
        if number_buf:
            expr_tokens.append(number_buf)
            number_buf = ""
        if tok in "+-*/=":
            expr_tokens.append(tok)
    if number_buf:
        expr_tokens.append(number_buf)
    return expr_tokens


def evaluate_simple(tokens: List[str]) -> Tuple[Optional[float], Optional[str]]:
    if not tokens:
        return None, "invalid expression"

    prec = {"+": 1, "-": 1, "*": 2, "/": 2}
    output: List[object] = []
    ops: List[str] = []
    prev_is_op = True

    for tok in tokens:
        if tok.isdigit():
            output.append(float(tok))
            prev_is_op = False
            continue
        if tok not in prec:
            return None, "invalid token"
        if tok == "-" and prev_is_op:
            output.append(0.0)
        while ops and ops[-1] in prec and prec[ops[-1]] >= prec[tok]:
            output.append(ops.pop())
        ops.append(tok)
        prev_is_op = True

    while ops:
        output.append(ops.pop())

    stack: List[float] = []
    for tok in output:
        if isinstance(tok, float):
            stack.append(tok)
            continue
        if len(stack) < 2:
            return None, "invalid expression"
        b = stack.pop()
        a = stack.pop()
        if tok == "+":
            stack.append(a + b)
        elif tok == "-":
            stack.append(a - b)
        elif tok == "*":
            stack.append(a * b)
        elif tok == "/":
            if abs(b) < 1e-8:
                return None, "division by zero"
            stack.append(a / b)
        else:
            return None, "invalid operator"

    if len(stack) != 1:
        return None, "invalid expression"
    return stack[0], None


def solve_tokens(tokens: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if "=" in tokens:
        idx = tokens.index("=")
        left = tokens[:idx]
        right = tokens[idx + 1 :]
        left_val, err = evaluate_simple(left)
        if err:
            return None, None, err
        if right:
            right_val, err = evaluate_simple(right)
            if err:
                return None, None, f"right side {err}"
            return left_val, right_val, None
        return left_val, None, None

    val, err = evaluate_simple(tokens)
    if err:
        return None, None, err
    return val, None, None


def load_symbol_image(
    path: Path,
    target_height: int,
    max_width_factor: float,
    ink_threshold: int,
    strip_bottom: bool,
    operator_height: float,
    is_operator: bool,
) -> Tuple[Image.Image, Image.Image]:
    img = Image.open(path).convert("L")
    mask = img.point(lambda v: 255 if v < ink_threshold else 0)
    bbox = mask.getbbox()
    if bbox:
        img = img.crop(bbox)
        mask = mask.crop(bbox)

    if strip_bottom:
        img, mask = strip_bottom_line(img, mask)

    w, h = img.size
    if h == 0:
        raise RuntimeError(f"Invalid image size: {path}")
    max_width = max(1, int(round(target_height * max_width_factor)))
    desired_height = target_height
    if is_operator:
        desired_height = max(10, int(round(target_height * operator_height)))

    scale = desired_height / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w > max_width:
        scale = max_width / float(w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

    if new_h < target_height:
        canvas = Image.new("L", (new_w, target_height), 255)
        mask_canvas = Image.new("L", (new_w, target_height), 0)
        y = (target_height - new_h) // 2
        canvas.paste(img, (0, y))
        mask_canvas.paste(mask, (0, y))
        img, mask = canvas, mask_canvas
    return img, mask


def render_equation(
    tokens: List[str],
    class_paths: Dict[str, List[Path]],
    target_height: int,
    spacing: int,
    max_width_factor: float,
    ink_threshold: int,
    strip_bottom: bool,
    operator_height: float,
) -> Image.Image:
    samples: List[Tuple[Image.Image, Image.Image]] = []
    for tok in tokens:
        cls = TOKEN_TO_CLASS.get(tok)
        if cls is None:
            continue
        sample_path = random.choice(class_paths[cls])
        img, mask = load_symbol_image(
            sample_path,
            target_height=target_height,
            max_width_factor=max_width_factor,
            ink_threshold=ink_threshold,
            strip_bottom=strip_bottom,
            operator_height=operator_height,
            is_operator=is_operator_class(cls),
        )
        samples.append((img, mask))

    total_w = sum(img.size[0] for img, _ in samples)
    total_w += spacing * max(0, len(samples) - 1)
    canvas = Image.new("L", (total_w, target_height), 255)

    x = 0
    for img, mask in samples:
        canvas.paste(img, (x, 0), mask)
        x += img.width + spacing
    return canvas


def stack_equations(
    eq_images: List[Image.Image],
    pad_x: int,
    pad_y: int,
    row_gap: int,
) -> Image.Image:
    max_w = max(img.width for img in eq_images)
    total_h = sum(img.height for img in eq_images) + row_gap * (len(eq_images) - 1)
    canvas = Image.new("L", (max_w + pad_x * 2, total_h + pad_y * 2), 255)
    y = pad_y
    for img in eq_images:
        x = pad_x
        canvas.paste(img, (x, y))
        y += img.height + row_gap
    return canvas


def random_number(min_val: int, max_val: int) -> int:
    return random.randint(min_val, max_val)


def generate_equation_tokens(max_value: int) -> List[str]:
    op = random.choice(OPS)
    if op == "*":
        a = random_number(1, 12)
        b = random_number(1, 12)
    elif op == "/":
        b = random_number(1, 9)
        result = random_number(0, 12)
        a = b * result
    elif op == "-":
        a = random_number(0, max_value)
        b = random_number(0, max_value)
        if b > a:
            a, b = b, a
    else:
        a = random_number(0, max_value)
        b = random_number(0, max_value)

    tokens = list(str(a)) + [op] + list(str(b)) + ["="]
    return tokens


def generate_dataset(
    out_dir: Path,
    class_paths: Dict[str, List[Path]],
    args: argparse.Namespace,
) -> List[dict]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[dict] = []

    for idx in range(args.images):
        eq_count = random.randint(1, args.max_eq)
        eq_images: List[Image.Image] = []
        eq_entries: List[dict] = []
        for _ in range(eq_count):
            tokens = generate_equation_tokens(args.max_result)
            eq_img = render_equation(
                tokens,
                class_paths,
                target_height=args.height,
                spacing=args.spacing,
                max_width_factor=args.max_width_factor,
                ink_threshold=args.ink_threshold,
                strip_bottom=args.strip_bottom,
                operator_height=args.operator_height,
            )
            eq_images.append(eq_img)
            expr_tokens = combine_tokens(tokens)
            left_val, right_val, err = solve_tokens(expr_tokens)
            value = None
            if err is None and right_val is None:
                value = left_val
            eq_entries.append(
                {
                    "tokens": tokens,
                    "expression": "".join(expr_tokens),
                    "value": value,
                }
            )
        full = stack_equations(eq_images, args.pad_x, args.pad_y, args.row_gap)
        filename = f"eq_{idx:04d}.png"
        rel_path = Path("images") / filename
        full.save(images_dir / filename)
        metadata.append({"image": str(rel_path), "equations": eq_entries})

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def values_match(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def evaluate_dataset(
    metadata: List[dict], images_root: Path, recognizer: Recognizer
) -> dict:
    total_eq = 0
    expr_match = 0
    result_match = 0
    count_match = 0
    image_all_correct = 0
    total_images = len(metadata)

    for entry in metadata:
        img_path = images_root / entry["image"]
        img = Image.open(img_path)
        gt_equations = entry["equations"]
        total_eq += len(gt_equations)

        preds = []
        for eq_box, pairs in segment_equations_with_boxes(img):
            labels = []
            for _box, crop in pairs:
                label, _conf = recognizer.predict(crop)
                if label is None:
                    continue
                labels.append(label)
            pred_tokens = combine_tokens(labels)
            preds.append(
                {
                    "expression": "".join(pred_tokens),
                    "tokens": pred_tokens,
                }
            )

        count_ok = len(preds) == len(gt_equations)
        if count_ok:
            count_match += 1
        image_ok = count_ok and len(gt_equations) > 0

        for idx, gt in enumerate(gt_equations):
            if idx >= len(preds):
                image_ok = False
                continue
            pred = preds[idx]
            if pred["expression"] == gt["expression"]:
                expr_match += 1
            else:
                image_ok = False

            gt_value = gt.get("value")
            pred_tokens = pred["tokens"]
            pred_left, pred_right, pred_err = solve_tokens(pred_tokens)
            if pred_err is None and pred_right is None and gt_value is not None:
                if values_match(pred_left, gt_value):
                    result_match += 1

        if image_ok:
            image_all_correct += 1

    report = {
        "images": total_images,
        "equations": total_eq,
        "image_count_match": count_match,
        "image_count_match_rate": count_match / total_images if total_images else 0.0,
        "image_all_correct": image_all_correct,
        "image_all_correct_rate": image_all_correct / total_images
        if total_images
        else 0.0,
        "expression_exact": expr_match,
        "expression_exact_rate": expr_match / total_eq if total_eq else 0.0,
        "result_match": result_match,
        "result_match_rate": result_match / total_eq if total_eq else 0.0,
        "bad_json": 0,
    }
    return report


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    data_root = Path("data")
    class_paths = load_class_paths(data_root)

    metadata = generate_dataset(out_dir, class_paths, args)

    if Path(DEFAULT_MODEL_PATH).is_file():
        recognizer = Recognizer.load(DEFAULT_MODEL_PATH)
    else:
        recognizer, _acc, _loaded = Recognizer.load_or_train("data", DEFAULT_MODEL_PATH)

    report = evaluate_dataset(metadata, out_dir, recognizer)
    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Synthetic evaluation complete")
    for key in [
        "images",
        "equations",
        "image_count_match",
        "image_count_match_rate",
        "image_all_correct",
        "image_all_correct_rate",
        "expression_exact",
        "expression_exact_rate",
        "result_match",
        "result_match_rate",
        "bad_json",
    ]:
        print(f"{key}: {report[key]}")


if __name__ == "__main__":
    main()
