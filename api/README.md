# Ukážka FastAPI

Tento priečinok pridáva FastAPI obal a malý webový demo klient, ktorý ho volá.

## Inštalácia

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r api\requirements.txt
```

## Spustenie

```powershell
py -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Otvorte http://127.0.0.1:8000 v prehliadači a použite demo.

## Použitie API

- POST /solve (multipart súbor s názvom poľa `file` s PNG/JPG)
- GET /health
