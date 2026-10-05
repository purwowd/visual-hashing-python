FROM python:3.12-slim-bookworm

EXPOSE 8000

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Shared libraries needed by skia-python on Linux
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libegl1 libexpat1 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
RUN adduser -u 666 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/', timeout=3)"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
