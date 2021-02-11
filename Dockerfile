FROM python:3.8-alpine

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run.py
ENV FLASK_SERVER_NAME=0.0.0.0:5000
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN apk update && apk add --no-cache \
	linux-headers \
	g++ \
	bash \
	libffi-dev \
	musl-dev \
	build-base \
	musl-dev \
	python3-dev \
	zeromq-dev \
	jpeg-dev \
	zlib-dev \
	freetype-dev \
	libjpeg-turbo-dev

ADD . /app
WORKDIR /app

# Install pytorch 1.7 - not available in world repo
RUN python3 -m pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apk del libffi-dev \
	musl-dev \
	build-base \
	musl-dev \
	python3-dev \
	zeromq-dev \
	jpeg-dev \
	zlib-dev \
	freetype-dev \
	libjpeg-turbo-dev

EXPOSE 5000

COPY . /app
CMD python run.py

