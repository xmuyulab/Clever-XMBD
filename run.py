import argparse
import numpy as np
import pandas as pd
import os
from Clever.dataset import *
from Clever.model import *

parser = argparse.ArgumentParser(description='Run predictions for unlabelled cells')
parser.add_argument("-data", type=str, help="path to input data file", default=None)
parser.add_argument("-outdir", type=str, help="directory to save output results", default='./result/')
parser.add_argument("-ncuda", type=int, help="CUDA device index to use", default=0)
parser.add_argument("-k", type=int, help="number of training iterations", default=5)
parser.add_argument("-wp", type=float, help="weight of posotive risk", default=5)
parser.add_argument("-ungated", type=str, help="name assigned to ungated cells", default='unlabelled', nargs='+')
parser.add_argument("-label", type=str, help="name of the column in the metadata that stores the gating labels", default=None)


args = parser.parse_args()

if __name__ == "__main__":

    data_path = args.data
    ncuda = args.ncuda
    iterations = args.k
    w_p = args.wp
    random_seed = 1
    neg_cell_type = args.ungated

    # Define paths for saving results and models
    save_path  = args.outdir
    model_path = save_path+"/models/"
    result_path = save_path+"/results/"

    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    if os.path.exists(model_path)==False:
        os.mkdir(model_path)
    if os.path.exists(result_path)==False:
        os.mkdir(result_path)

    pre, ext = os.path.splitext(data_path)
    dataname = pre.split('/')[-1]
    prefix = "Clever_"+dataname+"_k"+str(iterations)+"_wp"+str(w_p)

    # Load the dataset, split into labeled and unlabeled data, and retrieve train and test indices
    (x_labeled, y_labeled), x_unlabeled, train_index_list, test_index_list, \
        label_num_transform, num_label_transform = load_dataset(
            data_path, neg_cell_type,
            lib_name=args.label,
            k=iterations,
            random_seed = random_seed)

    predict,prob = Clever(x_labeled , y_labeled, x_unlabeled, train_index_list, batchsize = 256, w_p = w_p, \
                          iterations = iterations, random_seed = random_seed, modelpath = model_path, \
                          ncuda = ncuda,prefix = prefix+"_model_k")

    x_total = pd.concat([x_labeled,x_unlabeled],axis = 0)

    result =pd.DataFrame({"cell.id":x_total.index, "predict":pd.Series(predict).map(num_label_transform)})
    proportions = result['predict'].value_counts(normalize=True).reset_index()
    proportions.columns = ['celltype', 'proportion']

    result.to_csv(result_path+prefix+"_result.csv",index = False)
    proportions.to_csv(result_path+prefix+"_proportion.csv",index = False)