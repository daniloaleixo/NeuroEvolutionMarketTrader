FROM python:3.7

# Install ta-lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xvf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure --prefix=/usr && make
RUN cd ta-lib && make install && ldconfig

# Install dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get install -y \
        gcc \
        build-essential \
        zlib1g-dev \
        wget \
        unzip \
        cmake \
        python3-dev \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libhdf5-dev \
    && apt-get clean

# Caching C lib
RUN pip install --user --upgrade pip
RUN pip install numpy --extra-index-url https://www.piwheels.org/simple

RUN pip install Flask \
    && pip install keras==2.2.4 \
    && pip install tensorflow==1.15 \
    && pip install TA-lib \
    && pip install gym \
    && pip install keras_metrics \
    && pip install backtrader \
    && pip install pandas \
    && pip install matplotlib \
    && pip install tqdm

# EXPOSE 5000

WORKDIR /app
ADD ./cnn_prediction /app/cnn_prediction
ADD ./core /app/core
ADD ./GA /app/GA
ADD ./misc /app/misc
ADD ./saved_models /app/saved_models
ADD ./utils /app/utils
ADD ./config* /app/
ADD ./app.py /app/app.py


ENV CONFIG_FILE=config-EURUSD-30M_v3.0141.json


ENTRYPOINT ["python"]
CMD ["app.py"]


# docker build -t ga-server -f docker/server/Dockerfile .
# docker run --rm -p 5000:5000  ga-server