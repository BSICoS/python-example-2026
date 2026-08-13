FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
# 1. Clonamos el repositorio dentro de la carpeta /challenge
RUN git clone https://github.com/dr-you-group/PROPHECG-Age-Single.git /challenge/PROPHECG-Age-Single \
	&& git -C /challenge/PROPHECG-Age-Single checkout 3629293ccc0726c1e747654ddee458f779970971

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt