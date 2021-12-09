FROM tensorflow/tensorflow:2.4.1-gpu

# Install graphviz
RUN apt-get update
RUN apt-get install graphviz nano -y

# Install extras
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# Copy POPNAS folder into the container
WORKDIR /exp
COPY . .

CMD ["bash"]