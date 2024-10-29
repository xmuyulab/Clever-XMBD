from Clever.Conf_MPU_loss import Conf_MPULoss
from Clever.dataset import make_kfold_dataloader
from torch.optim import lr_scheduler
from Clever.utils import *
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


def training(model, class_num,train_dataloader, modelpath,lr = 0.05, num_epoch = 200, w_p = 5, ncuda = 0, prefix = "model"):
    """
    Train model using Conf-MPU loss.
    
    Args:
        model: The neural network model to train.
        class_num: Number of classes.
        train_dataloader: DataLoader for the training data.
        modelpath: Path to save the trained model.
        lr: Learning rate.
        num_epoch: Number of epochs to train.
        w_p: Weight for positive risk.
        prefix: Prefix for the model save file name.
    
    Returns:
        model: Trained model.
    """
    min_epoch = 30
    last_epoch_loss = 0

    optimizers = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler=lr_scheduler.StepLR(optimizers, step_size=30, gamma=0.1)
    loss_funcs = Conf_MPULoss(int(class_num-1), w_p, ncuda)

    if torch.cuda.is_available():
        print("use gpu")
        model.cuda(ncuda)

    if os.path.exists(modelpath)==False:
        os.makedirs(modelpath)

    # Obtain initial training data from train_dataloader
    data_list = []
    label_list = []
    for data, labels in train_dataloader:
        data_list.append(data)
        label_list.append(labels)
    data = torch.cat(data_list)
    labels = torch.cat(label_list)

    # training loop
    n = 0
    last_epoch_loss = 100
    for epoch in range(num_epoch):
        model.train()
        step_loss = list()

        train_dataset = list(zip(data, labels))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True, drop_last=True)
        
        for x, t in train_dataloader:
            x, t = x.cuda(ncuda), t.cuda(ncuda)
            
            optimizers.zero_grad()

            h = model(x)
            loss = loss_funcs(h,t)

            loss.backward()
            optimizers.step()

            step_loss.append(loss.item())

        # Self-training
        if abs(last_epoch_loss - np.mean(step_loss)) < 0.01:
            # Predict unlabeled data and generate pseudo-labels
            model.eval()
            count_confidence = 0
            with torch.no_grad():
                new_data_list = []
                new_label_list = []
                for x, t in train_dataloader:
                    x, t = x.cuda(ncuda), t.cuda(ncuda)
                    h = model(x)
                    probs = torch.softmax(h, dim=1)
                    confidence, predicted_labels = torch.max(probs, dim=1)
                    
                    # Update only for unlabeled data
                    unlabeled_mask = (t == class_num - 1)
                    high_confidence_mask = confidence > 0.99
                    combined_mask = unlabeled_mask & high_confidence_mask

                    # Update labels
                    t[combined_mask] = predicted_labels[combined_mask]
                    count_confidence += len(predicted_labels[combined_mask])
                    new_data_list.append(x.cpu())
                    new_label_list.append(t.cpu())

                data = torch.cat(new_data_list)
                labels = torch.cat(new_label_list)

        scheduler.step()
        last_epoch_loss = np.mean(step_loss)

        # Update weight for positive risk
        if epoch==4:
            loss_funcs = Conf_MPULoss(int(class_num-1), 1, ncuda)

        # early-stopping
        if epoch >= min_epoch:
            if count_confidence == 0:
                torch.save(model.state_dict(), modelpath+'/'+str(prefix)+'.pt')
                n += 1
            else:
                n = 0

            if n==5:
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


def Clever(x_labeled , y_labeled, x_unlabeled,train_index_list=None, batchsize = 256, w_p = 5, iterations = 5, random_seed = 777, modelpath = "./",ncuda = 0, prefix = None):
    """
    Main function to run the Clever workflow for training and prediction.

    Args:
        x_labeled: Labeled feature data.
        y_labeled: Labeled target data.
        x_unlabeled: Unlabeled feature data.
        train_index_list: List of training indices for k models.
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
    
    if iterations>1:
        for iteration in range(iterations):
            print("Start iteration {}".format(iteration+1))

            # Prepare data
            x_train, y_train = make_kfold_dataloader((x_labeled, y_labeled,x_unlabeled),train_index_list, iteration,balanced = False)

            #train loader
            train_dataset=list(zip(x_train,y_train))
            train_loader = torch.utils.data.DataLoader(train_dataset, batchsize, shuffle=True)

            #build model
            input_dim = x_train[0].size
            class_num = len(np.unique(y_train))
            model = ThreeLayerPerceptron(input_dim, class_num)

            # # Base classification model training
            model = training(model, class_num, train_loader, modelpath, prefix = prefix+str(iteration), w_p= w_p, ncuda = ncuda)
    else:
        raise NotImplementedError("The iteration number should be larger than 1")
    
    # prediction
    x_total = pd.concat([x_labeled,x_unlabeled],axis=0)
    predict, prob = get_average_predict(iterations,modelpath,x_total,feature,class_num,prefix)

    return predict, prob