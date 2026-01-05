import os
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw

from Recognizer import (
    DEFAULT_MODEL_PATH,
    Recognizer,
    preprocess_to_mnist28_image,
    segment_equations_with_boxes,
)

CANVAS_SIZE = 900
BRUSH_RADIUS = 9

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Classifier (scikit-learn)")

        self.recognizer, acc, loaded = Recognizer.load_or_train(
            "data", DEFAULT_MODEL_PATH
        )
        self.solve_job = None
        self.auto_solve_delay_ms = 350

        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", highlightthickness=1
        )
        self.canvas.grid(row=0, column=0, rowspan=8, padx=(0, 10))

        self.pred_var = tk.StringVar(value="Prediction: -")
        self.conf_var = tk.StringVar(value="Confidence: -")
        if loaded:
            self.info_var = tk.StringVar(
                value=f"Loaded model: {DEFAULT_MODEL_PATH}"
            )
        else:
            self.info_var = tk.StringVar(
                value=f"MNIST+Ops MLPClassifier | acc={acc:.3f}"
            )
        self.expr_var = tk.StringVar(value="Expression: -")
        self.result_var = tk.StringVar(value="Result: -")

        ttk.Label(frame, textvariable=self.pred_var, font=("Segoe UI", 16)).grid(row=0, column=1, sticky="w")
        ttk.Label(frame, textvariable=self.conf_var).grid(row=1, column=1, sticky="w")
        ttk.Label(frame, textvariable=self.info_var).grid(row=2, column=1, sticky="w", pady=(0, 10))
        ttk.Label(frame, textvariable=self.expr_var).grid(row=3, column=1, sticky="w")
        ttk.Label(frame, textvariable=self.result_var).grid(row=4, column=1, sticky="w", pady=(0, 10))

        ttk.Button(frame, text="Predict", command=self.predict).grid(row=5, column=1, sticky="we")
        ttk.Button(frame, text="Solve", command=self.solve).grid(row=6, column=1, sticky="we", pady=(6, 0))
        ttk.Button(frame, text="Clear", command=self.clear).grid(row=7, column=1, sticky="we")

        # Backing image for preprocessing
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

    def on_down(self, e):
        self.last_x, self.last_y = e.x, e.y
        self._dot(e.x, e.y)

    def on_move(self, e):
        if self.last_x is None:
            self.last_x, self.last_y = e.x, e.y
        w = BRUSH_RADIUS * 2
        self.canvas.create_line(self.last_x, self.last_y, e.x, e.y,
                                fill="black", width=w, capstyle=tk.ROUND, smooth=True)
        self.draw.line([self.last_x, self.last_y, e.x, e.y], fill=0, width=w)
        self._dot(e.x, e.y)
        self.last_x, self.last_y = e.x, e.y

    def on_up(self, e):
        self.last_x = self.last_y = None
        self._schedule_auto_solve()

    def _dot(self, x, y):
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.draw = ImageDraw.Draw(self.image)
        if self.solve_job is not None:
            self.root.after_cancel(self.solve_job)
            self.solve_job = None
        self.pred_var.set("Prediction: -")
        self.conf_var.set("Confidence: -")
        self.expr_var.set("Expression: -")
        self.result_var.set("Result: -")

    def _schedule_auto_solve(self):
        if self.solve_job is not None:
            self.root.after_cancel(self.solve_job)
        self.solve_job = self.root.after(
            self.auto_solve_delay_ms, self._maybe_auto_solve
        )

    def _maybe_auto_solve(self):
        self.solve_job = None
        equations = segment_equations_with_boxes(self.image)
        if not equations:
            return
        for _eq_box, pairs in equations:
            for _box, crop in pairs:
                label, _conf = self.recognizer.predict(crop)
                if label == "=":
                    self.solve()
                    return

    def predict(self):
        label, conf = self.recognizer.predict(self.image)
        if label is None:
            self.pred_var.set("Prediction: (draw something)")
            self.conf_var.set("Confidence: -")
            return
        self.pred_var.set(f"Prediction: {label}")
        self.conf_var.set(f"Confidence: {conf:.3f}")

    def solve(self):
        self.canvas.delete("segments")
        self.canvas.delete("equations")
        self.canvas.delete("result_text")
        equations = segment_equations_with_boxes(self.image)
        if not equations:
            self.expr_var.set("Expression: (draw something)")
            self.result_var.set("Result: -")
            return

        expr_strings = []
        result_strings = []
        for eq_box, pairs in equations:
            x0, y0, x1, y1 = eq_box
            self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="blue", width=2, tags="equations"
            )

            tokens = []
            labeled_boxes = []
            # self._save_segments(pairs)
            for (sx0, sy0, sx1, sy1), crop in pairs:
                self.canvas.create_rectangle(
                    sx0, sy0, sx1, sy1, outline="red", width=2, tags="segments"
                )
                label, _conf = self.recognizer.predict(crop)
                if label is not None:
                    tokens.append(label)
                    labeled_boxes.append(((sx0, sy0, sx1, sy1), label))

            expr_tokens = self._combine_tokens(tokens)
            if not expr_tokens:
                expr_strings.append("(could not read)")
                result_strings.append("-")
                continue

            expr_str = "".join(expr_tokens)
            expr_strings.append(expr_str)

            left_val, right_val, status = self._solve_tokens(expr_tokens)
            if status is not None:
                result_strings.append(status.replace("Result: ", ""))
                continue

            if right_val is None:
                result_text = self._format_number(left_val)
            else:
                match = abs(left_val - right_val) < 1e-6
                flag = "OK" if match else "Mismatch"
                left_str = self._format_number(left_val)
                right_str = self._format_number(right_val)
                result_text = f"{left_str} | Right: {right_str} ({flag})"
            result_strings.append(result_text)

            eq_symbol_box = None
            for box, label in labeled_boxes:
                if label == "=":
                    eq_symbol_box = box
                    break
            if eq_symbol_box is not None and left_val is not None:
                sx0, sy0, sx1, sy1 = eq_symbol_box
                result_str = self._format_number(left_val)
                self.canvas.create_text(
                    sx1 + 10,
                    (sy0 + sy1) / 2,
                    text=result_str,
                    anchor="w",
                    font=("Segoe UI", 18, "bold"),
                    fill="blue",
                    tags="result_text",
                )

        if len(expr_strings) == 1:
            self.expr_var.set(f"Expression: {expr_strings[0]}")
            self.result_var.set(f"Result: {result_strings[0]}")
        else:
            self.expr_var.set("Expressions: " + " | ".join(expr_strings))
            self.result_var.set("Results: " + " | ".join(result_strings))

    def _combine_tokens(self, tokens):
        expr_tokens = []
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

    def _solve_tokens(self, tokens):
        if "=" in tokens:
            idx = tokens.index("=")
            left = tokens[:idx]
            right = tokens[idx + 1 :]
            left_val, err = self._evaluate_simple(left)
            if err:
                return None, None, f"Result: {err}"
            if right:
                right_val, err = self._evaluate_simple(right)
                if err:
                    return None, None, f"Result: right side {err}"
                return left_val, right_val, None
            return left_val, None, None

        val, err = self._evaluate_simple(tokens)
        if err:
            return None, None, f"Result: {err}"
        return val, None, None

    def _evaluate_simple(self, tokens):
        if not tokens:
            return None, "invalid expression"

        prec = {"+": 1, "-": 1, "*": 2, "/": 2}
        output = []
        ops = []
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

        stack = []
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

    def _format_number(self, val):
        if val is None:
            return "-"
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        return f"{val:.4f}"

    def _save_segments(self, pairs):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("segments_debug", stamp)
        raw_dir = os.path.join(out_dir, "raw")
        proc_dir = os.path.join(out_dir, "processed")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)
        for idx, (_box, crop) in enumerate(pairs):
            raw_path = os.path.join(raw_dir, f"seg_{idx:02d}.png")
            crop.save(raw_path)
            proc = preprocess_to_mnist28_image(crop)
            if proc is None:
                continue
            proc_path = os.path.join(proc_dir, f"seg_{idx:02d}.png")
            proc.save(proc_path)

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
