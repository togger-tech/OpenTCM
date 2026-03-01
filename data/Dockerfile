FROM python:3.12-slim

# Instala dependências como root (no build time — sem problemas de permissão)
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Cria usuário não-root com UID/GID passados via build-arg
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} appgroup && \
    useradd -u ${APP_UID} -g appgroup -s /bin/sh -M appuser

# Muda para usuário não-root
USER appuser

WORKDIR /app

CMD ["python", "-u", "data/TCMKG.py"]
