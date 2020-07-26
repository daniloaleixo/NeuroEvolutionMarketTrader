
FROM jupyter/scipy-notebook:latest

ENV GRANT_SUDO yes
USER root

# Install ta-lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xvf ta-lib-0.4.0-src.tar.gz && cd ta-lib && ./configure --prefix=/usr && make
RUN cd ta-lib && make install && ldconfig

RUN pip install --upgrade pip

RUN pip install keras==2.2.4
RUN pip install tensorflow==1.15
RUN pip install TA-lib
RUN pip install gym && pip install keras_metrics && pip install backtrader

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
