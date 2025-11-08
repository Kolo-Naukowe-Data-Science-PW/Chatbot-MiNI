FROM mambaorg/micromamba:1.5.8

COPY environment-linux.yml /tmp/environment.yml
RUN micromamba create -y -n app -f /tmp/environment.yml && micromamba clean -a -y

SHELL ["micromamba", "run", "-n", "app", "/bin/bash", "-lc"]

WORKDIR /app
COPY . /app


ENV PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPATH=/app/src
CMD ["micromamba", "run", "-n", "app", "python", "-c", "print('image ready')"]
