FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install transformers  lightgbm scikit-learn sentencepiece onnx onnxruntime-gpu pynvml psutil python-Levenshtein
COPY utils /opt/conda/lib/python3.8/site-packages/starter_kit/
#put all necessary code into utils


# The online submission environment requires running as a non-root user named aicrowd. If run as root, the system can pass public phase but will fail on private phase. I guess the system try to delete the file generated at public phase but fail due to no permission.
ENV USER aicrowd
ENV HOME /home/aicrowd
RUN groupadd --gid 1001 aicrowd
RUN useradd --comment "Default user" --create-home --gid 1001 --no-log-init --shell /bin/bash --uid 1001 aicrowd
USER aicrowd