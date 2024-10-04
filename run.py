import argparse
import numpy as np
import pandas as pd
import os
from dataset import *
from model import *


parser = argparse.ArgumentParser(description='run_unlabelled_cells_prediction')
parser.add_argument("-data", type=str, help="dataset name", default="IMC_data")
parser.add_argument("-datadir", type=str, help="data file directory", default='example')
parser.add_argument("-outdir", type=str, help="output result file directory", default='result')
parser.add_argument("-ncuda", type=int, help="No. cuda", default=0)
parser.add_argument("-i", type=int, help="iterations", default=5)
parser.add_argument("-wp", type=float, help="weight of posotive risk", default=5)
parser.add_argument("-ungated", type=str, help="name of ungated cells", default='unlabelled')

args = parser.parse_args()

if __name__ == "__main__":

    dataset = args.data
    ncuda = args.ncuda
    iterations = args.i
    w_p = args.wp
    random_seed = 1
    neg_cell_type = [args.ungated]
    neg_cell_type.append('Other')

    # Define paths for saving results and models
    save_path  = args.outdir
    data_path = args.datadir
    model_path = save_path+"/models"+"_i"+str(iterations)+"_wp"+str(w_p)+"/"
    result_path = save_path+"/result"+"_i"+str(iterations)+"_wp"+str(w_p)+"/"

    if os.path.exists(save_path)==False:
        os.mkdir(save_path)
    if os.path.exists(model_path)==False:
        os.mkdir(model_path)
    if os.path.exists(result_path)==False:
        os.mkdir(result_path)

    prefix = "Clever_"+dataset+"_r"+str(random_seed)+"_i"+str(iterations)+"_wp"+str(w_p)

    # Load the dataset, split into labeled and unlabeled data, and retrieve train and test indices
    (x_labeled, y_labeled),(x_unlabeled, y_unlabeled), train_index_list, test_index_list, \
        label_num_transform, num_label_transform = load_dataset(
            dataset, data_path, neg_cell_type,
            lib_name='layer_1_gated', # Cell types based on specific gating criteria
            ref_lib_name = 'celltype', # Reference cell types provided by the data provider, used for comparison. If not available, you can set it to be the same as 'lib_name'.
            k=iterations,
            random_seed = random_seed)

    predict,prob = Clever(x_labeled , y_labeled, x_unlabeled, train_index_list, batchsize = 256, w_p = w_p, \
                          iterations = iterations, random_seed = random_seed, modelpath = model_path, \
                          ncuda = ncuda,prefix = prefix+"_model_i")

    x_total = pd.concat([x_labeled,x_unlabeled],axis = 0)
    y_total = np.append(y_labeled,y_unlabeled)
    result =pd.DataFrame({"sample.id":x_total.index,"reference":pd.Series(y_total).map(num_label_transform),"predict":pd.Series(predict).map(num_label_transform)})
    result_prob = pd.DataFrame(prob,index = x_total.index,columns=num_label_transform.values())

    result.to_csv(result_path+prefix+"_result.csv",index = False)
    result_prob.to_csv(result_path+prefix+"_prob.csv")
