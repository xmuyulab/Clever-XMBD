import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import os
np.random.seed(1)


def get_data_path(dataset_name,datadir):
    """
    Args:
        dataset_name: The name of the dataset.
        datadir: The directory where the dataset is located.

    Returns:
        str: The full path to the dataset file.
    """
    data_path = os.path.join(datadir, '.'.join([dataset_name, 'h5ad']))

    return data_path


def label_to_num(label, label_num_transform):
    """
    Converts categorical labels to numerical labels using a given mapping.

    Args:
        label: Array of categorical labels.
        label_num_transform: Dictionary mapping categorical labels to numerical labels.

    Returns:
        np.ndarray: Array of numerical labels.
    """
    num_label = np.array(pd.Series(label).map(label_num_transform))

    return num_label


def num_to_label(num_label, num_label_transform):
    """
    Converts numerical labels back to categorical labels using a given mapping.

    Args:
        num_label: Array of numerical labels.
        num_label_transform: Dictionary mapping numerical labels to categorical labels.

    Returns:
        np.ndarray: Array of categorical labels.
    """
    label = np.array(pd.Series(num_label).map(num_label_transform))

    return label


def get_data(dataset_name,datadir,neg_cell_type,lib_name,ref_lib_name):
    """
    Loads and processes the dataset, splitting it into labeled and unlabeled sets.
    """

    # Get the path to the data file based on the dataset name and directory.
    data_path = get_data_path(dataset_name,datadir)
    
    # Load the data from the file and extract the cell expression and cell type information.
    test_data = sc.read_h5ad(data_path)

    cell_exp = pd.DataFrame(test_data.X,index=test_data.obs.index,columns=test_data.var_names)
    cell_library = test_data.obs[lib_name]
    cell_library_ref = test_data.obs[ref_lib_name]
    
    # Split the data into labeled and unlabeled based on gating labels.
    cell_exp_labeled = cell_exp[cell_library.isin(neg_cell_type)==False]
    cell_exp_unlabeled = cell_exp.drop(index = cell_exp_labeled.index,inplace=False)
    
    cell_library_labeled = cell_library[cell_exp_labeled.index]
    cell_library_unlabeled = cell_library[cell_exp_unlabeled.index]
    cell_libraryref_unlabeled = cell_library_ref[cell_exp_unlabeled.index]
  
    # Sort the cell types and create a dictionary to map them to numerical labels.
    celltype_list = np.unique(cell_library)
    celltype_list_sort = list(set(celltype_list)-set(neg_cell_type))
    celltype_list_sort.sort()
    num_label_transform =  {idx:label for idx,label in enumerate(np.array(celltype_list_sort))}
    num_label_transform[len(celltype_list_sort)] = "Unknown"
    celltype_list_sort.extend(neg_cell_type)
    
    label_num_transform =  {label:idx for idx,label in enumerate(np.array(celltype_list_sort))}

    cell_labeled_target = label_to_num(cell_library_labeled, label_num_transform)
    cell_unlabeled_target = label_to_num(cell_library_unlabeled, label_num_transform)
    cell_unlabeled_ref = label_to_num(cell_libraryref_unlabeled, label_num_transform)

    print(label_num_transform)
    print(num_label_transform)
    return (cell_exp_labeled,cell_labeled_target),(cell_exp_unlabeled,cell_unlabeled_target),cell_unlabeled_ref, label_num_transform, num_label_transform


def rename_negative_class(y_train, y_test,neg_cell_type):
    """
    Rename test targets to negative class.

    Args:
        y_train: Array of training targets.
        y_test: Array of test targets.
        neg_cell_type: Negative cell types list.

    Returns:
    """
    neg_class_num = len(neg_cell_type)
    all_class_num = len(np.unique(np.concatenate((np.array(y_train),np.array(y_test)))))
    pos_class_num = all_class_num - neg_class_num
    y_test_bin = np.array(y_test, dtype=np.int64)
    y_test_bin[y_test >= pos_class_num] = pos_class_num
    return y_test_bin


def load_dataset(dataset_name,datadir,neg_cell_type,lib_name,ref_lib_name,k):
    """
    Loads the dataset and prepares it for K-fold cross-validation.

    Args:
        dataset_name: The name of the dataset.
        datadir: The directory where the dataset is located.
        neg_cell_type: List of cell types considered as unalbelled.
        lib_name: The column name for gating cell types.
        ref_lib_name: The column name for reference cell types.
        k: Number of folds for K-fold cross-validation.

    Returns:
    """
    #generate kfold index
    def creat_kfold(k,x_unlabel):
        kf = KFold(n_splits=k, random_state=None)
        kf_generator = kf.split(x_unlabel)

        train_index_list = []
        test_index_list = []

        for test_index, train_index in kf_generator:
            train_index_list.append(train_index)
            test_index_list.append(test_index)
        
        return train_index_list, test_index_list
    
    #split labeled and unlabelled data
    (x_labeled, y_labeled), (x_unlabeled, y_unlabeled), y_unlabeled_ref, \
        label_num_transform, num_label_transform = get_data(dataset_name,datadir,neg_cell_type,lib_name,ref_lib_name)
    
    y_unlabeled = rename_negative_class(y_labeled, y_unlabeled,neg_cell_type)
    
    #shuffle unlabelled set
    perm = np.random.permutation(len(y_unlabeled))
    x_unlabeled, y_unlabeled = x_unlabeled.iloc[perm], y_unlabeled[perm]
    y_unlabeled_ref = y_unlabeled_ref[perm]
    
    train_index_list, test_index_list = creat_kfold(k,x_unlabeled)
    
    return (x_labeled, y_labeled),(x_unlabeled, y_unlabeled_ref), train_index_list, test_index_list,label_num_transform, num_label_transform



def make_kfold_dataloader(dataset, train_index_list, iteration, random_seed, balanced = False):
    """
    Creates data loaders for K-fold cross-validation.

    Args:
        dataset: Contains labeled and unlabeled cell expressions.
        train_index_list: List of training indices for K-fold cross-validation.
        iteration: Current fold iteration.
        random_seed: Seed for random number generator.
        balanced: Whether to balance the dataset using SMOTE.

    Returns:
        Training and validation data and targets.
    """

    def split_unlabel(x_unlabeled, train_index_list, iteration):
        """
        Splits the unlabeled data for the current fold iteration.
        """
    
        x_unlabel_train = x_unlabeled.iloc[train_index_list[iteration],:]

        return x_unlabel_train
    
    def make_pu_dataset_from_multiclass_dataset(x_labeled, y_labeled, x_unlabel_train, balanced = False):
        """
        Creates a positive-unlabeled (PU) dataset from a multiclass dataset.
        """
        negative = int(len(np.unique(y_labeled)))
        x_labeled, y_labeled = np.asarray(x_labeled, dtype=np.float32), np.asarray(y_labeled, dtype=np.int64)
        x_unlabel_train = np.asarray(x_unlabel_train, dtype=np.float32)
        
        if balanced:
            model_smote = SMOTE()
            x_labeled, y_labeled = model_smote.fit_resample(x_labeled, y_labeled)
            
        n_u = x_unlabel_train.shape[0]
        
        
        x = np.asarray(np.concatenate((x_labeled, x_unlabel_train), axis=0), dtype=np.float32)
        y = np.asarray(np.concatenate((np.array(y_labeled), (np.ones(n_u)*int(negative)).astype(int))), dtype=np.int64)
        perm = np.random.permutation(len(y)) # shuffle randomly
        x, y = x[perm], y[perm]
        return x, y
                 
    (x_labeled, y_labeled, x_unlabeled) = dataset

    # Split labeled data into training and validation sets
    x_labeled_train, x_labeled_vali, y_labeled_train, y_labeled_vali = train_test_split(x_labeled, y_labeled,test_size=0.2,random_state=random_seed)

    x_unlabel_train = split_unlabel(x_unlabeled,train_index_list,iteration)

    x_train, y_train = make_pu_dataset_from_multiclass_dataset(x_labeled_train, y_labeled_train, x_unlabel_train, balanced)

    x_vali, y_vali = np.asarray(x_labeled_vali, dtype=np.float32), np.asarray(y_labeled_vali, dtype=np.int64)
    
    print("training:{}".format(x_train.shape))
    print("validation:{}".format(x_labeled_vali.shape))
    
    return x_train, y_train, x_vali, y_vali