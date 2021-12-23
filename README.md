# unsupervised-depth-estimation
Unsupervised single-shot depth estimation with perceptual reconstruction loss

Implementation of a framework for fully unsupervised single-view depth estimation as proposed in:

## Installation

```
#from github
git clone https://github.com/anger-man/unsupervised-depth-estimation
cd unsupervised-depth-estimation
conda env create --name tf_2.2.0 --file=environment.yml
conda activate tf_2.2.0
```
## Usage

Model architectures and training parameters can be set in **config_file.ini**.
Then:
```
python train.py --direc path_to_projet_folder
```

