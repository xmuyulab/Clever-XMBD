from Conf_MPU_loss import Conf_MPULoss
from dataset import make_kfold_dataloader
from utils import *
import torch
from torch import nn
import os, sys
import numpy as np
import random
import pandas as pd


class ThreeLayerPerceptron(nn.Module):
    def __init__(self, INPUT_size, class_num, hidden_dim = 32):
        super(ThreeLayerPerceptron, self).__init__()
        self.INPUT_size = INPUT_size
        self.class_num = class_num
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_num),
        )
        
    def forward(self, x):        
        out = self.encoder(x)
        return out


def training(model, class_num,train_dataloader, Xvali,Yvali,modelpath,lr = 0.01, num_epoch = 100, w_p = 4, ncuda = 0, prefix = "model"):
    """
    Train model using Conf-MPU loss.
    
    Args:
        model: The neural network model to train.
        class_num: Number of classes.
        train_dataloader: DataLoader for the training data.
        Xvali: Validation data features.
        Yvali: Validation data labels.
        modelpath: Path to save the trained model.
        lr: Learning rate.
        num_epoch: Number of epochs to train.
        w_p: Weight for positive risk.
        prefix: Prefix for the model save file name.
    
    Returns:
        model: Trained model.
    """
    min_epoch = 50
    min_f1 = 0

    model.train()
    optimizers = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)

    if torch.cuda.is_available():
        print("use gpu")
        model.cuda(ncuda)

    loss_funcs = Conf_MPULoss(int(class_num-1), w_p, ncuda)

    if os.path.exists(modelpath)==False:
        os.makedirs(modelpath)

    # training loop
    f1_vali = list()
    for epoch in range(num_epoch):
        step_loss = list()

        # train
        for x, t in train_dataloader:
            x, t = x.cuda(ncuda), t.cuda(ncuda)
            
            optimizers.zero_grad()

            h = model(x)
            loss = loss_funcs(h,t)

            loss.backward()
            optimizers.step()

            step_loss.append(loss.item())

        f1_vali.append(evaluate_val(model,Xvali,Yvali,ncuda))


        # early-stopping
        if epoch >= min_epoch:
            if f1_vali[epoch] >= min_f1:
                min_f1 = f1_vali[epoch]
                torch.save(model.state_dict(), modelpath+'/'+str(prefix)+'.pt')
                n = 0
            else:
                n += 1

            if n==10:
                break

        sys.stdout.write('\r')
        sys.stdout.write(' Epoch %3d \t Conf-MPU_loss: %.6f \n' %(epoch, np.mean(step_loss)))
        sys.stdout.flush()
         
    model.load_state_dict(torch.load(modelpath+'/'+str(prefix)+'.pt'))
    print("\n Model training finish!")

    return model


def get_average_predict(iterations,model_path,x_unlabeled,feature,class_num,prefix=None):
    """
    Get average predictions from the model over multiple iterations.

    Args:
        iterations: Number of iterations/models to average.
        model_path: Path to the saved model files.
        x_unlabeled: Unlabeled/whole data to predict.
        feature: List of feature names.
        class_num: Number of classes.
        prefix: Prefix for the model file names.
    
    Returns:
        predict: Predicted class labels.
        prob: Predicted class probabilities.
    """
    x_unlabeled = x_unlabeled.T
    x_unlabeled = x_unlabeled.reindex(feature)
    x_unlabeled = x_unlabeled.T
    input_dim = len(feature)
    X_unlabeled = np.asarray(x_unlabeled, dtype=np.float32)
    
    model = ThreeLayerPerceptron(input_dim, class_num)

    for iteration in range(iterations):
        model.load_state_dict(torch.load(model_path+'/'+prefix+str(iteration)+'.pt'))
        model.eval()
        output = model(torch.from_numpy(X_unlabeled))
        output_softmax= torch.nn.functional.softmax(output, 1)
        output_softmax = output_softmax.unsqueeze(2)
        if iteration ==0:
            output_softmax_total = output_softmax
        else:
            output_softmax_total = torch.cat((output_softmax_total,output_softmax),dim=2)
            
    output_softmax_total_mean = torch.mean(output_softmax_total,dim=2)        
    _,predict = torch.max(output_softmax_total_mean,1)
    predict=Variable(predict,requires_grad=False).cpu().numpy()
    prob=Variable(output_softmax_total_mean,requires_grad=False).cpu().numpy()
    
    return predict, prob


def Clever(x_labeled , y_labeled, x_unlabeled,train_index_list=None, batchsize = 256, w_p = 10, iterations = 5, random_seed = 777, modelpath = "./",ncuda = 0, prefix = None):
    """
    Main function to run the Clever workflow for training and prediction.

    Args:
        x_labeled: Labeled feature data.
        y_labeled: Labeled target data.
        x_unlabeled: Unlabeled feature data.
        train_index_list: List of training indices for k-fold cross-validation.
        batchsize: Batch size for training.
        w_p: Weight for positive risk in Conf-MPU loss.
        iterations: Number of training iterations/models.
        modelpath: Path to save the models.
        prefix: Prefix for the model file names.
    
    Returns:
        predict: Predicted class labels for the desired dataset (unlabeled/whole data).
        prob: Predicted class probabilities for the desired dataset (unlabeled/whole data).
    """
    # set random seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    feature = list(x_labeled.columns)
    
    if iterations>0:
        for iteration in range(iterations):
            print("Start iteration {}".format(iteration+1))

            # Prepare data for k-fold cross-validation
            x_train,y_train,x_vali, y_vali = make_kfold_dataloader((x_labeled, y_labeled,x_unlabeled),train_index_list, iteration,random_seed,balanced = False)

            #train loader
            train_dataset=list(zip(x_train,y_train))
            train_loader = torch.utils.data.DataLoader(train_dataset, batchsize, shuffle=True)

            #build model
            input_dim = x_train[0].size
            class_num = len(np.unique(y_train))
            model = ThreeLayerPerceptron(input_dim, class_num)

            # # Base classification model training
            model = training(model, class_num, train_loader, x_vali, y_vali, modelpath, prefix = prefix+str(iteration), w_p= w_p, ncuda = ncuda)
    else:
        raise NotImplementedError("The iteration number should be larger than 1")
    
    # prediction
    x_total = pd.concat([x_labeled,x_unlabeled],axis=0)
    predict, prob = get_average_predict(iterations,modelpath,x_total,feature,class_num,prefix)

    return predict, prob