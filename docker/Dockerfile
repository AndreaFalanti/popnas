FROM tensorflow/tensorflow:2.6.0rc0-gpu

# Install extras
COPY requirements.txt /requirements.txt
RUN /bin/bash -c "pip install -r requirements.txt"

# Copy POPNAS folder into the container
WORKDIR /exp
COPY . .

#CMD ["python", "run.py", "-b", "5", "-k", "2", "-e", "1"]
CMD ["bash"]
