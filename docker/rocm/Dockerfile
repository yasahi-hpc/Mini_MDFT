# SPDX-FileCopyrightText: (C) The Mini-MDFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

ARG BASE=rocm/dev-ubuntu-22.04:5.6-complete
FROM $BASE

RUN apt-get update && apt-get install -y \
        kmod \
        wget \
        ccache \
        git \
        file \
        hipfft \
        rocfft \
        libfftw3-dev \
        $ADDITIONAL_PACKAGES \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV FFTWDIR "/usr"

ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH /opt/rocm/hipfft/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH /opt/rocm/hip/:/opt/rocm/:$CMAKE_PREFIX_PATH

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

WORKDIR /work
CMD ["bash"]
