# SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

ARG BASE=intel/oneapi-basekit:2024.0.1-devel-ubuntu20.04
FROM $BASE

RUN apt-get update && apt-get install -y \
        bc \
        wget \
        git \
        ccache \
        libfftw3-dev \
        $ADDITIONAL_PACKAGES \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV FFTWDIR "/usr"

RUN git config --global --add safe.directory "*"

# Install newer CMake manually
ARG CMAKE_VERSION=3.25.2
ARG build_dir=/tmp/build

ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    mkdir -p ${build_dir} && mkdir -p ${CMAKE_DIR} \
    cd ${build_dir} && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_SCRIPT} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    cd ${HOME} && rm -rf ${build_dir}
ENV PATH=${CMAKE_DIR}/bin:$PATH

WORKDIR /work
CMD ["bash"]
