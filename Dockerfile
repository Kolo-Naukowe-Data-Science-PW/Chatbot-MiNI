FROM mambaorg/micromamba:1.5.8

COPY environment-linux.yml /tmp/environment.yml
RUN micromamba create -y -n app -f /tmp/environment.yml && micromamba clean -a -y

SHELL ["micromamba", "run", "-n", "app", "/bin/bash", "-lc"]

# Needed by pandas read_excel / to_markdown (ingest_manual_xlsx, describe_files).
# Separate layer so the cached conda/pip environment above is not rebuilt.
RUN /opt/conda/envs/app/bin/python -m pip install --no-cache-dir openpyxl tabulate

WORKDIR /app
COPY . /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=/app/src \
    LD_LIBRARY_PATH=/opt/conda/envs/app/lib:$LD_LIBRARY_PATH

CMD ["micromamba", "run", "-n", "app", "python", "-c", "print('image ready')"]
