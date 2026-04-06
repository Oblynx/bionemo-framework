# Uncomment to use the latest TE from the NGC registry for debugging changes with latest TE.
# FROM gitlab-master.nvidia.com/dl/transformerengine/transformerengine:main-pytorch-py3-base
FROM nvcr.io/nvidia/pytorch:26.03-py3

# FIXME: Fix for "No such file or directory: /workspace/TransformerEngine"
#  Remove once bug has been addressed in the nvidia/pytorch container.
RUN rm -f /usr/local/lib/python*/dist-packages/transformer_engine-*.dist-info/direct_url.json

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=/workspace/requirements.txt \
    PIP_CONSTRAINT= pip install -r /workspace/requirements.txt

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
USER ubuntu
RUN uv tool install pre-commit --with pre-commit-uv --force-reinstall
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
# Link the claude CLI from the VS Code extension on each shell start.
# Extensions are installed after all devcontainer lifecycle hooks, so ~/.bashrc
# is the earliest reliable place to do this.
RUN printf '\n# Link claude CLI from VS Code extension into PATH\n_b=$(find ~/.vscode-server/extensions -name "claude" -path "*/native-binary/claude" 2>/dev/null | head -1); [ -x "$_b" ] && ln -sf "$_b" ~/.local/bin/claude; unset _b\n' >> /home/ubuntu/.bashrc
