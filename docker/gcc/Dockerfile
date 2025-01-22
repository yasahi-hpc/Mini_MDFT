# SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

ARG BASE=ubuntu:22.04
FROM $BASE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        git \
        bc \
        ccache \
        libfftw3-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory "*"

# Install newer CMake manually
ARG CMAKE_VERSION=3.23.2
ARG build_dir=/tmp/build

ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    mkdir -p ${build_dir} && mkdir -p ${CMAKE_DIR} \
    cd ${build_dir} && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_SCRIPT} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    cd ${HOME} && rm -rf ${build_dir}
ENV PATH=${CMAKE_DIR}/bin:$PATH
ENV FFTWDIR "/usr"

WORKDIR /work
CMD ["bash"]