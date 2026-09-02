# Container image for the CFB Prediction Dashboard.
# Used by Hugging Face Spaces (sdk: docker, app_port 7860) and works
# unchanged on Railway / Render / Fly.io / plain `docker run`.

FROM python:3.11-slim

# LightGBM and XGBoost need the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires the container to run as a non-root user (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /home/user/app

# Install dependencies first so the layer caches across code changes.
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# App code + the prebuilt data / model artifacts (baked in by the deploy
# workflow; see docs/DEPLOY.md).
COPY --chown=user:user . .

EXPOSE 7860
# HF Spaces sets $PORT; default to 7860 for local runs.
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0"]
