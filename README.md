# Clever
Clever represents **C**el**L** abundanc**E** quantification using multi-positi**V**e unlab**E**led lea**R**ning. It is a semi-automated, deep learning-based computational framework designed to effectively identify and quantify cells from single-cell multiplexed imaging and proteomic datasets. Clever not only accurately classifies cells but also preserves biological interpretability.

## Installation
### Dependencies
```
Python 3.8.18
Pytorch 1.7.1
pandas
numpy
scikit-learn
tqdm
anndata
scanpy
fcsparser
```

a. Set up a python environment (conda virtual environment is recommended).

```shell
conda create -n clever python=3.8
git clone https://github.com/xmuyulab/Clever-XMBD.git
cd Clever
conda activate clever
```

b. Install Pytorch.

```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

c. Install required python packages.

```shell
pip install -r requirements.txt
```

## Data preparation and preprocessing

Our method requires a `.h5ad` file as input. The input AnnData should contain `adata.X` with arcsinh-transformed marker expression data, `adata.var` with marker information, and `adata.obs` with cell observations. To prepare your data, you can either prepare a `.h5ad` file directly or provide an `.fcs` file containing flow cytometry data, along with an optional `.csv` file with cell observation data.

**Note:** To create your own data using an FCS file, please refer to `example.ipynb`.

## Usage

Our model utilizes the labeled portion of the dataset for training purposes and then predicts the entire dataset (including the unlabelled portion). Use `data` to specify the dataset name, `datadir` for the input data directory, and `outdir` for the directory where the results will be stored. The `ungated` parameter indicates the labels of cells that were not assigned during the gating process.
```
python run.py -data IMC_data -datadir example -outdir result -ungated unlabelled -i x -wp x
```
- `i` (iterations): This hyperparameter specifies the number of training models.

- `wp` (weight of positive risk): This hyperparameter specifies the weight for positive risk in the loss function.