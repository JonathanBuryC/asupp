FROM registry.apps.caas-int-hp.automation.edf.fr/pud-usine-dap-grp-hp/python-minimal-uv:3.11.11

WORKDIR ${APP_HOME}

ENV NETRC=/netrc \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/python/cpython-3.11.13-linux-x86_64-gnu \
    UV_NATIVE_TLS=1 \
    UV_HTTP_TIMEOUT=90 \
    UV_HTTP_RETRIES=10

# Étape 1 : installation des dépendances
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=secret,id=nexusauth,dst=/netrc,uid=1001 \
    uv sync --frozen --no-install-project --no-dev --no-python-downloads

# Étape 2 : copie uniquement les répertoires nécessaires
COPY --chown=1001 src ${APP_HOME}/src
COPY --chown=1001 main.py ${APP_HOME}/
COPY --chown=1001 run.sh ${APP_HOME}/
COPY --chown=1001 config ${APP_HOME}/config
COPY --chown=1001 scripts ${APP_HOME}/scripts
COPY --chown=1001 livraisons-standalone-* ${APP_HOME}/
COPY --chown=1001 pyproject.toml uv.lock ${APP_HOME}/

# Étape 3 : installation du projet
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=nexusauth,dst=/netrc,uid=1001 \
    uv sync --frozen --no-dev && \
    rm -rf /root/.cache /tmp/* || true

CMD ["uv", "run", "--no-dev", "--no-sync", "python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "5001"]
