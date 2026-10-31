# One image, two processes (see docker-compose.yml): the FastAPI backend and the Streamlit UI.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY analyst ./analyst
COPY analyst_mcp ./analyst_mcp
COPY api ./api
COPY ui ./ui
COPY eval ./eval
COPY data ./data
COPY pyproject.toml ./

# Run as an unprivileged user: generated code runs in a subprocess of this process.
RUN useradd --create-home analyst && mkdir -p outputs logs uploads && chown -R analyst /app
USER analyst

EXPOSE 8000 8501
# API by default; the UI service in docker-compose overrides the command.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
