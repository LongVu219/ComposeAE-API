FROM python:3.6.8

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app

CMD bert-serving-start -model_dir bert-model/bert_uncase -num_worker=8 & \
    python -W ignore inference.py