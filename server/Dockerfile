FROM python:3.8-slim-buster
RUN apt-get update

WORKDIR /app

COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "run.py"]

EXPOSE 5555
EXPOSE 5550
