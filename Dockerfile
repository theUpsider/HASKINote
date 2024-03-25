FROM python:3.9.12-bullseye
WORKDIR /code

COPY ./tloen ./
COPY ./main.py ./
#COPY ./.env.prod ./.env
ADD requirements.txt /code

# pip install -r requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]