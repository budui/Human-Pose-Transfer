FROM tensorflow/tensorflow:1.4.1-gpu-py3

# for user in China, you can use PYPI mirror.
# docker build -t hpt_evaluate . --build-arg PIP_PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"
ARG PIP_PYPI="https://pypi.org/simple/"
ARG PIP_INSTALL="pip --no-cache-dir install -i ${PIP_PYPI} --upgrade"

RUN $PIP_INSTALL tqdm scikit-image