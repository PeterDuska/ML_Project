from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from Recognizer import DEFAULT_MODEL_PATH, Recognizer, segment_equations_with_boxes

WEB_DIR = Path(__file__).resolve().parent / 'web'

app = FastAPI(title='Math Recognizer API', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

recognizer: Optional[Recognizer] = None


@app.on_event('startup')
def _load_model() -> None:
    global recognizer
    recognizer, _acc, _loaded = Recognizer.load_or_train('data', DEFAULT_MODEL_PATH)


def _combine_tokens(tokens: List[str]) -> List[str]:
    expr_tokens: List[str] = []
    number_buf = ''
    for tok in tokens:
        if tok.isdigit():
            number_buf += tok
            continue
        if number_buf:
            expr_tokens.append(number_buf)
            number_buf = ''
        if tok in '+-*/=':
            expr_tokens.append(tok)
    if number_buf:
        expr_tokens.append(number_buf)
    return expr_tokens


def _evaluate_simple(tokens: List[str]) -> Tuple[Optional[float], Optional[str]]:
    if not tokens:
        return None, 'neplatný výraz'

    prec = {'+': 1, '-': 1, '*': 2, '/': 2}
    output: List[object] = []
    ops: List[str] = []
    prev_is_op = True

    for tok in tokens:
        if tok.isdigit():
            output.append(float(tok))
            prev_is_op = False
            continue
        if tok not in prec:
            return None, 'neplatný token'
        if tok == '-' and prev_is_op:
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
            return None, 'invalid expression'
        b = stack.pop()
        a = stack.pop()
        if tok == '+':
            stack.append(a + b)
        elif tok == '-':
            stack.append(a - b)
        elif tok == '*':
            stack.append(a * b)
        elif tok == '/':
            if abs(b) < 1e-8:
                return None, 'delenie nulou'
            stack.append(a / b)
        else:
            return None, 'neplatný operátor'

    if len(stack) != 1:
        return None, 'neplatný výraz'
    return stack[0], None


def _solve_tokens(tokens: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if '=' in tokens:
        idx = tokens.index('=')
        left = tokens[:idx]
        right = tokens[idx + 1 :]
        left_val, err = _evaluate_simple(left)
        if err:
            return None, None, err
        if right:
            right_val, err = _evaluate_simple(right)
            if err:
                return None, None, f'pravá strana {err}'
            return left_val, right_val, None
        return left_val, None, None

    val, err = _evaluate_simple(tokens)
    if err:
        return None, None, err
    return val, None, None


def _format_number(val: Optional[float]) -> str:
    if val is None:
        return '-'
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f'{val:.4f}'


def _result_payload(
    left_val: Optional[float],
    right_val: Optional[float],
    err: Optional[str],
) -> dict:
    if err:
        return {'status': 'error', 'message': err}
    if right_val is None:
        return {'status': 'ok', 'value': _format_number(left_val)}
    match = left_val is not None and abs(left_val - right_val) < 1e-6
    return {
        'status': 'ok',
        'left': _format_number(left_val),
        'right': _format_number(right_val),
        'match': match,
    }


def _to_int_list(box: Tuple[int, int, int, int]) -> List[int]:
    return [int(v) for v in box]


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}


@app.post('/solve')
async def solve(file: UploadFile = File(...)) -> dict:
    if recognizer is None:
        raise HTTPException(status_code=503, detail='model nie je načítaný')

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail='prázdny upload')

    try:
        img = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f'neplatný obrázok: {exc}'
        ) from exc

    equations = []
    for eq_box, pairs in segment_equations_with_boxes(img):
        symbols = []
        for box, crop in pairs:
            label, conf = recognizer.predict(crop)
            if label is None:
                continue
            symbols.append(
                {'box': _to_int_list(box), 'label': label, 'confidence': float(conf)}
            )

        labels = [s['label'] for s in symbols]
        expr_tokens = _combine_tokens(labels)
        expr_str = ''.join(expr_tokens) if expr_tokens else ''
        left_val, right_val, err = _solve_tokens(expr_tokens)
        equations.append(
            {
                'equation_box': _to_int_list(eq_box),
                'symbols': symbols,
                'expression_tokens': expr_tokens,
                'expression': expr_str,
                'result': _result_payload(left_val, right_val, err),
            }
        )

    return {'equation_count': len(equations), 'equations': equations}


if WEB_DIR.is_dir():
    app.mount('/', StaticFiles(directory=str(WEB_DIR), html=True), name='web')
