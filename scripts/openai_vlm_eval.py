import argparse
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# put your openAI_API_KEY
#OPENAI_API_KEY = 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic equation images with an OpenAI VLM."
    )
    parser.add_argument(
        "--data-dir",
        default="synthetic_equations",
        help="Folder with images/ and metadata.jsonl",
    )
    parser.add_argument("--metadata", default="metadata.jsonl")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--out", default="vlm_report.json")
    parser.add_argument("--preds", default="vlm_predictions.jsonl")
    return parser.parse_args()


def load_metadata(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def build_prompt() -> str:
    return (
        "You are an OCR system for handwritten math. "
        "Return JSON only in the format: "
        "{\"equations\":[\"8-3=\",\"2+5=\"]}. "
        "Rules: output equations in top-to-bottom order, left-to-right within a line. "
        "Use digits 0-9 and operators + - * / = only. "
        "If you see a division sign (dot-line-dot), output '/'. "
        "If you see 'x' for multiplication, output '*'. "
        "No spaces, no extra text."
    )


def extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_value(expr: str) -> Optional[float]:
    expr = expr.strip()
    if not expr.endswith("="):
        return None
    expr = expr[:-1]
    for op in ["+", "-", "*", "/"]:
        if op in expr:
            parts = expr.split(op)
            if len(parts) != 2:
                return None
            try:
                left = int(parts[0])
                right = int(parts[1])
            except ValueError:
                return None
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                if right == 0:
                    return None
                return left / right
    return None


def values_match(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def request_vlm(client: OpenAI, model: str, image_path: Path) -> Optional[dict]:
    prompt = build_prompt()
    b64 = image_to_base64(image_path)
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                    },
                ],
            }
        ],
        temperature=0,
    )
    data = extract_json(resp.output_text or "")
    return data


def normalize_equations(items: List[str]) -> List[str]:
    cleaned = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned.append(item.replace(" ", ""))
    return cleaned


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / args.metadata
    images_root = data_dir

    items = load_metadata(metadata_path)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    # If you set OPENAI_API_KEY above, pass it explicitly here.
    client = OpenAI(api_key=OPENAI_API_KEY)
    #client = OpenAI()
    preds_path = data_dir / args.preds
    out_path = data_dir / args.out

    total_eq = 0
    expr_match = 0
    result_match = 0
    image_count_match = 0
    image_all_correct = 0
    bad_json = 0

    with preds_path.open("w", encoding="utf-8") as f:
        for idx, entry in enumerate(items):
            img_path = images_root / entry["image"]
            gt_equations = entry["equations"]
            total_eq += len(gt_equations)

            try:
                resp = request_vlm(client, args.model, img_path)
            except Exception as exc:
                f.write(
                    json.dumps(
                        {
                            "image": entry["image"],
                            "error": str(exc),
                        }
                    )
                    + "\n"
                )
                bad_json += 1
                continue

            if resp is None or "equations" not in resp:
                bad_json += 1
                pred_equations: List[str] = []
            else:
                pred_equations = normalize_equations(resp.get("equations", []))

            if len(pred_equations) == len(gt_equations):
                image_count_match += 1

            all_correct = True
            for i, gt in enumerate(gt_equations):
                if i >= len(pred_equations):
                    all_correct = False
                    continue
                if pred_equations[i] == gt["expression"]:
                    expr_match += 1
                else:
                    all_correct = False

                pred_value = parse_value(pred_equations[i])
                gt_value = gt.get("value")
                if values_match(pred_value, gt_value):
                    result_match += 1

            if all_correct and len(gt_equations) > 0:
                image_all_correct += 1

            f.write(
                json.dumps(
                    {
                        "image": entry["image"],
                        "gt": [g["expression"] for g in gt_equations],
                        "pred": pred_equations,
                    }
                )
                + "\n"
            )

            if args.delay > 0:
                time.sleep(args.delay)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(items)} images")

    report: Dict[str, float] = {
        "images": len(items),
        "equations": total_eq,
        "expression_exact": expr_match,
        "expression_exact_rate": expr_match / total_eq if total_eq else 0.0,
        "result_match": result_match,
        "result_match_rate": result_match / total_eq if total_eq else 0.0,
        "image_count_match": image_count_match,
        "image_count_match_rate": image_count_match / len(items) if items else 0.0,
        "image_all_correct": image_all_correct,
        "image_all_correct_rate": image_all_correct / len(items) if items else 0.0,
        "bad_json": bad_json,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("VLM evaluation complete")
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
