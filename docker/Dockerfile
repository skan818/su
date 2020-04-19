FROM tensorflow/tensorflow:2.2.0rc3-gpu-jupyter
# Grab a list of current requirements & install them
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# Make docker act as default image
WORKDIR /tf
EXPOSE 8888

RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
