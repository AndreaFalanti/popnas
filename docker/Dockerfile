FROM tensorflow/tensorflow:2.7.0-gpu

# Install needed system dependencies
RUN apt update && apt install graphviz nano git-all python3-tk -y

# Install extra python dependencies
COPY requirements.txt /requirements.txt
RUN /bin/bash -c "pip install -r requirements.txt"

# Copy POPNAS folder into the container
WORKDIR /exp
COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/exp"
ENV TFDS_DATA_DIR="tensorflow_datasets"

CMD ["bash"]
