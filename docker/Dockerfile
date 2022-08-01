FROM tensorflow/tensorflow:2.7.3-gpu

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

# Install poetry (v1.0.0 produce a smaller docker image, without detectable drawbacks. Watch out for potential problems)
RUN pip install --ignore-installed "poetry==1.0.0"

# Copy only requirements to cache them in docker layer
WORKDIR /exp
COPY poetry.lock pyproject.toml /exp/

# Install the dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev
#RUN poetry export --dev --without-hashes --no-interaction --no-ansi -f requirements.txt -o requirements.txt
#RUN pip install --prefix=/runtime --force-reinstall -r requirements.txt

# Copying src folder into /exp (source code for executing POPNASv2)
COPY src/ .

# solve import problems with scripts and add specific folder for tfds mounts
ENV PYTHONPATH="${PYTHONPATH}:/exp"
ENV TFDS_DATA_DIR="tensorflow_datasets"

CMD ["bash"]
