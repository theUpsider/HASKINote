FROM continuumio/miniconda3
WORKDIR /code


COPY ./environment.yml ./

# Create new env
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "haskinote", "/bin/bash", "-c"]

COPY ./tloen ./
COPY ./main.py ./
COPY ./.env.prod ./.env

RUN echo "source activate haskinote" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "haskinote", "python3", "main.py"]