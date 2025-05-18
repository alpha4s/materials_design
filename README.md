# Materials Design Toolkit

A modular, end-to-end pipeline for AI-driven materials property prediction and design.  
Built around a conditional VAE + predictor ensemble, with nearest-neighbor lookup and gradient/BO-based inversion.

## Features
- **Predict**: formula → numeric properties  
- **Invert**: target properties → nearest known materials  
- **Sample**: random latent samples → candidate materials  
- **Design**: gradient descent in latent space for desired properties  
- **BO**: Bayesian-optimization in latent space  

## Installation
MUST RUN THE DATA-PREP FILE TO GET THE MATERIALS_DATASET.CSV FILE FOR THE OTHER FILES TO WORK
git clone https://github.com/alpha4s/materials_design.git
cd materials_design
pip install -r requirements.txt
# materials_design
Machine Learning model made to predict element properties and suggest new chemical compositions
