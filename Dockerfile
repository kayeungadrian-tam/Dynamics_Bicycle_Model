FROM python:3.8-slim-buster
USER root

WORKDIR /app
RUN apt-get update

ENTRYPOINT ["python3"]

COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

EXPOSE 8050