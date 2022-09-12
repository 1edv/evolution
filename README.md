

# The evolution, evolvability and engineering of gene regulatory DNA


[![Paper DOI : https://doi.org/10.1038/s41586-022-04506-6](https://badgen.net/badge/Nature%20DOI/10.1038%2Fs41586-022-04506-6/F96854)](https://doi.org/10.1038/s41586-022-04506-6) &nbsp; [![Star](https://img.shields.io/github/stars/1edv/evolution.svg?logo=github&style=social)](https://github.com/1edv/evolution) &nbsp; [![Follow](https://badgen.net/badge/twitter/Eeshit%20Dhaval%20Vaishnav)](https://twitter.com/i/user/1349259546) &nbsp; [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://1edv.github.io/evolution/)



## Quickstart
Use the live app now. __No downloads__. __No installation.__ 👇 
<p align = 'center'>
<a href='https://1edv.github.io/evolution/'><img align="center" src="https://img.icons8.com/nolan/96/artificial-intelligence.png"/></a>  

</p>

[comment]: <> (<a href=https://1edv.github.io/evolution/><img src="https://img.icons8.com/nolan/96/artificial-intelligence.png"/></a>) 

[![App demonstration|635x380](demo.gif)](https://evolution-app-vbxxkl6a7a-ue.a.run.app/)


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
1. After installing docker and pulling the latest image as described in the first two steps above, run the following on a terminal :
```bash
docker run -it --rm --entrypoint /bin/bash edv123456789/evolution_app

python
```

2. In the python shell, run :
```python
from app_aux import *

model_condition = 'Glu' #or, 'SC_Ura'

model, _ , __ = load_model(model_condition) 

model.summary()
```

You have now loaded our ```tensorflow.keras``` model. You may use this as is for downstream computations as described in the manuscript or adapt it for your application (e.g. transfer learning). 

To exit the python shell and the docker container, simply press ```Ctrl+D``` twice.

## All data and code
All data and code is also available in an interactive, fully functional form as a CodeOcean capsule (just press 'Reproducible Run' on CodeOcean to reproduce results) at :
<p align = 'center'>
<a href='https://codeocean.com/capsule/8020974/tree/v1'><img align="center" src="https://img.icons8.com/nolan/96/artificial-intelligence.png"/></a>  
</p>

## Gene Expression Prediction DREAM Challenge 2022
For more details, pleae see the <a href='https://synapse.org/#!Synapse:syn28469146/wiki/617075'>DREAM Challenge webpage</a> 

#### Timeline
  - Registration Open: April 2022
  - Launch: May 2, 2022
  - Webinar: May 2, 2022
  - Leaderboard Submissions for challenge opens: May 16, 2022
  - Final Submissions for the challenge: July 30 2022
  - Winners Announced: August 15, 2022
  - Results presented at TBD
  - Access challenge data TBD

## Reference 

_The evolution, evolvability and engineering of gene regulatory DNA, <b>Nature</b> 2022._

Eeshit Dhaval Vaishnav<sup>\*§</sup>,  Carl G. de Boer<sup>\*§</sup>, Jennifer Molinet, Moran Yassour, Lin Fan, Xian Adiconis, Dawn A. Thompson, Joshua Z. Levin, Francisco A. Cubillos, Aviv Regev<sup>§</sup>. 

DOI : https://doi.org/10.1038/s41586-022-04506-6




