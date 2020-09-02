FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
  git \
  python3.7 \
  python3-pip \
  python3-dev

RUN pip3 install --upgrade pip

RUN pip install flask

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
