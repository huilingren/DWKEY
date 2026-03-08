### Software dependencies

Users should install the following packages first, which will install in about 5 minutes on a machine with the recommended specs. The versions of software are, specifically:
```
absl-py==2.1.0
astunparse==1.6.3
cachetools==5.5.2
charset-normalizer==3.4.1
Cython==3.0.12
flatbuffers==25.2.10
gast==0.4.0
google-auth==2.38.0
google-auth-oauthlib==1.0.0
google-pasta==0.2.0
grpcio==1.70.0
h5py==3.11.0
idna==3.10
igraph==0.11.8
importlib_metadata==8.5.0
keras==2.13.1
libclang==18.1.1
Markdown==3.7
MarkupSafe==2.1.5
networkx==3.1
numpy==1.24.4
oauthlib==3.2.2
opt_einsum==3.4.0
packaging==24.2
pandas==2.0.3
protobuf==4.25.6
pyasn1==0.6.1
pyasn1_modules==0.4.1
python-dateutil==2.9.0.post0
python-igraph==0.11.8
pytz==2025.1
requests==2.32.3
requests-oauthlib==2.0.0
rsa==4.9
scipy==1.10.1
six==1.17.0
tensorboard==2.13.0
tensorboard-data-server==0.7.2
tensorflow==2.13.1
tensorflow-estimator==2.13.0
tensorflow-io-gcs-filesystem==0.34.0
termcolor==2.4.0
texttable==1.7.0
tqdm==4.67.1
typing_extensions==4.5.0
tzdata==2025.1
urllib3==2.2.3
Werkzeug==3.0.6
wrapt==1.17.2
zipp==3.20.2
```
# Installation Guide

## Instructions
1. First install all the above required packages, which are contained in the requirements.txt file
```
pip install -r requirements.txt
```
2. Make all the file
```
python setup.py build_ext -i
```

## Typical install time

# Reproduction instructions

# Instructions to run

Train the model,  
``` 
CUDA_VISIBLE_DEVICES=0 python train.py                
```
Train the model,  
``` 
CUDA_VISIBLE_DEVICES=0 python train.py     
