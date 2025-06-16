# FROM ubuntu

# WORKDIR /src


# RUN apt-get update
# RUN apt-get -y install python3
# RUN apt-get -y install python3-sklearn


# COPY Machine_Learning_training.ipynb ./Machine_Learning_training.ipynb


# CMD ["python3", "Machine_Learning_training.ipynb"]


FROM python:3.12

LABEL authors="Vinh"

WORKDIR /src

COPY support_vector.pkl /src/support_vector.pkl
COPY vectorizer.pkl /src/vectorizer.pkl
COPY server.py /src/server.py
COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]

