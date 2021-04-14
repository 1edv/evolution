

# The evolution, evolvability and engineering of gene regulatory DNA


[![Paper DOI : https://doi.org/10.1101/2021.02.17.430503](https://zenodo.org/badge/DOI/10.1101/2021.02.17.430503.svg)](https://doi.org/10.1101/2021.02.17.430503) &nbsp; [![Star](https://img.shields.io/github/stars/1edv/evolution.svg?logo=github&style=social)](https://github.com/1edv/evolution) &nbsp; [![Follow](https://img.shields.io/twitter/follow/edv_tweets?style=social)](https://www.twitter.com/edv_tweets) &nbsp; [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/1edv/evolution/app/app.py)



## Quickstart
Use the live app now. __No downloads__. __No installation.__ 👇 
<p align = 'center'>
<a href='https://share.streamlit.io/1edv/evolution/app/app.py'><img align="center" src="https://img.icons8.com/nolan/96/artificial-intelligence.png"/></a>  

</p>

[comment]: <> (<a href=https://evolution-app-vbxxkl6a7a-uc.a.run.app/><img src="https://img.icons8.com/nolan/96/artificial-intelligence.png"/></a>) 

[![Example of live coding an app in Streamlit|635x380](demo.gif)](https://share.streamlit.io/1edv/evolution/app/app.py)


## Repository overview
The GitHub repository is organized into two directories : 
- ```app``` : A fully self-contained directory for running the interactive application to compute mutational robustness, evolvability vectors and expression.

- ```manuscript_code``` : The codebase corresponding to the manuscript. The organization of this directory is further described [here](manuscript_code/README.md).


## Run the app locally
If you wish to run the app on your local machine or cluster, 

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Run the following commands on a terminal :
```bash
docker pull edv123456789/evolution_app:latest

docker run --rm -d  -p 8501:8501/tcp edv123456789/evolution_app:latest
```
The app is now running and you can access it by navigating to [http://localhost:8501/](http://localhost:8501/) in your web browser. If running on a remote cluster, you may want to expose port ```8501``` using [ngrok](https://ngrok.com/).

## Using the model directly
1. After installing docker and pulling the latest image as described in the first two steps above, run the following command on a terminal :
```bash
docker run -it --rm --entrypoint /bin/bash edv123456789/evolution_app

python
```

2. In the python shell, run :
```python
from app_aux import *

model, _ , __ = load_model('Glu')

model.summary()
```

You have now loaded our ```tensorflow.keras``` model loaded. You may use this as is for downstream computations as described in the manuscript or adapt it for your application by transfer learning. 

To exit the docker container, simply press ```Ctrl+D``` twice.

## Reference and Data
DOI : https://doi.org/10.1101/2021.02.17.430503\
_A comprehensive fitness landscape model reveals the evolutionary history and future evolvability of eukaryotic cis-regulatory DNA sequences, biorXiv 2021._\
Eeshit Dhaval Vaishnav,  Carl G. de Boer,  Moran Yassour,  Jennifer Molinet, Lin Fan,  Xian Adiconis, Dawn A. Thompson,  Francisco A. Cubillos,  Joshua Z. Levin,  Aviv Regev. 



