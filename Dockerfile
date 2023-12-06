FROM continuumio/miniconda3
WORKDIR /code

COPY ./tloen ./
COPY ./main.py ./
# COPY ./.env.prod ./.env
COPY ./requirements.txt ./

# pip install -r requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]