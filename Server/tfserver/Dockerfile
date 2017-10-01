# Load the ubuntu base image
FROM ubuntu:latest

MAINTAINER Maxime De Bruyn <maximedebruyn@gmail.com>

# Install the dependencies needed for tensorflow serving.
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        mlocate \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        wget \
        && \
    apt-get clean && \
rm -rf /var/lib/apt/lists/*

# Not sure if needed. Think it's only for the client.curl -H "Content-Type: application/json" -X POST -d '{"sentence":"this is my sentence"}' http://0.0.0.0:5000/api/is_correct
RUN pip install mock grpcio

# Install tensorflow serving from apt-get
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

RUN apt-get update && apt-get install tensorflow-model-server

# Expose port 9000 as this is the port that will be used by the tf server.
EXPOSE 9000

# Actually run the server
ENTRYPOINT tensorflow_model_server --port=9000 --model_name=instacorrect --model_base_path=/app/data/models/

# When starting the container do not forget to ADD THE VOLUME.
# For example, you can start the container with the following command:
# docker run -p 9000:9000 -v user/Documents/Server/tfserver/models:/app/data/models maximedb/tensorflow-serving running_tf_server
