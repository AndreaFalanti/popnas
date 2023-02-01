FROM python:3.10 as poetry-build

ENV POETRY_VERSION=1.3.2

WORKDIR /poetry
COPY poetry.lock pyproject.toml /poetry/

# install poetry and generate the requirements.txt from the lock file
RUN pip install poetry==${POETRY_VERSION}
RUN poetry export --without-hashes --no-interaction --no-ansi -o requirements.txt


FROM tensorflow/tensorflow:2.8.4-gpu

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# Update CUDA GPG key
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Install needed system dependencies
RUN apt update && apt install graphviz nano -y

# Upgrade pip
RUN pip install --upgrade pip==22.3.1

# Copy only requirements to cache them in docker layer
WORKDIR /exp
COPY --from=poetry-build /poetry/requirements.txt /exp/

# Install the dependencies without creating a virtual environment (use system python env)
RUN pip install -r requirements.txt

# Copying src folder into /exp (source code for executing POPNAS)
COPY src/ .

# solve import problems with scripts and add specific folder for tfds mounts
ENV PYTHONPATH="${PYTHONPATH}:/exp"
ENV TFDS_DATA_DIR="datasets/tensorflow_datasets"

CMD ["bash"]
