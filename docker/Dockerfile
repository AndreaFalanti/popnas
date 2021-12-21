FROM tensorflow/tensorflow:2.7.0-gpu

# Install graphviz
RUN apt-get update
RUN apt-get install graphviz nano -y

# Install extras
COPY requirements.txt /requirements.txt
RUN /bin/bash -c "pip install -r requirements.txt"

# Copy POPNAS folder into the container
WORKDIR /exp
COPY . .

ENV PYTHONPATH="${PYTHONPATH}:/exp"

CMD ["bash"]
