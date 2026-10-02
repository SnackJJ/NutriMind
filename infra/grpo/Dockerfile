FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG MIMO_COMMIT=a2ad9f6160b03ff2d47e59832bfb6b289f37c917
ENV DEBIAN_FRONTEND=noninteractive UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv git build-essential libnuma1 ca-certificates \
    && apt-get clean
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
RUN uv venv --python python3.12 /opt/grpo
COPY infra/grpo/requirements.lock.txt /opt/requirements.lock.txt
RUN uv pip install --python /opt/grpo/bin/python -r /opt/requirements.lock.txt
RUN git clone https://github.com/XiaomiMiMo/verl.git /opt/mimo-verl \
    && git -C /opt/mimo-verl checkout --detach "$MIMO_COMMIT" \
    && uv pip install --no-deps --python /opt/grpo/bin/python /opt/mimo-verl
WORKDIR /workspace/NutriMind
COPY . .
RUN uv pip install --no-deps --python /opt/grpo/bin/python infra/grpo/vllm_plugin
ENV GRPO_VENV=/opt/grpo MIMO_VERL_ROOT=/opt/mimo-verl \
    NUTRIENV_ROOT=/opt/nutri-env-pin UV_BIN=/usr/local/bin/uv \
    PATH=/opt/grpo/bin:$PATH
# Mount the pinned NutriEnv checkout read-only and the model/data directory separately.
ENTRYPOINT ["bash", "scripts/run_grpo_v2.sh"]
CMD ["4090"]
