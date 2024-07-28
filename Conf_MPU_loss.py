import torch
import numpy as np

class Conf_MPULoss(torch.nn.Module):
    """wrapper of loss function for MPU learning"""

    def __init__(self, class_num , w_p=4, ncuda=0):
        super().__init__()
        self.class_num = class_num
        self.w_p = w_p  # weight of positive_risk
        self.ncuda = ncuda

    def forward(self, x, t):
        return conf_mpu_loss(x, t, self.class_num, w_p= self.w_p, ncuda=self.ncuda)
    


def conf_mpu_loss(x, t, class_num, w_p=4, ncuda=0):
    """wrapper of loss function for Conf-MPU
    Args:
        x: Outputs of model.
        t: True labels for the data.
        class_num: Number of positive classes.
        w_p: Weight for positive risk.
        
    Returns:
        x_out: Calculated loss.
    """


    # Calculate empirical priors for each class
    prior_emp = []
    N = len(t)
    for i in range(class_num):
        prior_emp.append( float((t == i).float().sum())/ float(N))
        
    
    def loss_per_class(x, t):
        # Compute the loss for each class.
        
        num_classes = x.size(1)
        h_softmax = torch.nn.functional.softmax(x, dim=1)
        t_one_hot = torch.nn.functional.one_hot(t, num_classes=num_classes).float()
        losses = -t_one_hot * torch.log(h_softmax + 1e-9)
        # Sum losses per class
        class_losses = losses.sum(dim=0)
        return class_losses
    

    def loss_per_class_neg(x, t):
        # Compute the negative loss for each class.

        num_classes = x.size(1)
        h_softmax = torch.nn.functional.softmax(x, dim=1)
        t_one_hot = torch.nn.functional.one_hot(t, num_classes=num_classes).float()
        
        p_class_neg = h_softmax[:, num_classes-1] # Consider only negative class
        loss_neg = []
        
        for i in range(num_classes-1):
            class_mask = (t == i)
            if class_mask.sum().float() == 0:
                loss_neg.append(torch.tensor(0.).cuda(ncuda))
            else:
                # Calculate negative loss for each positive class
                loss_neg.append((-t_one_hot[class_mask, i] * torch.log(p_class_neg[class_mask] + 1e-9)).sum())
        return loss_neg

    def conf_loss_per_class(x, t, threshold = 0.5):
        #  Compute the confident based loss for each class.
        
        num_classes = x.size(1)
        h_softmax = torch.nn.functional.softmax(x, dim=1)
        t_one_hot = torch.nn.functional.one_hot(t, num_classes=num_classes).float()
        loss_list = []
        for i in range(num_classes):
            if i < num_classes-1:
                mask = (h_softmax[:, i] > threshold) & (t==i)
            else:
                mask = torch.all(h_softmax <= threshold, dim=1) & (t==i) # Mask for predictions with all class probabilities below threshold

            num_selected = mask.sum().float()
            # If no samples meet the condition, add zero loss and continue
            if num_selected == 0:
                loss_list.append(torch.tensor(0.).cuda(ncuda))
                continue

            selected_t_one_hot = t_one_hot[mask]
            selected_h_softmax = h_softmax[mask]
            losses = -selected_t_one_hot[:, i] * torch.log(selected_h_softmax[:, -1] + 1e-9)
            if i < num_classes-1:
                # Divide by the corresponding probability
                losses /= selected_h_softmax[:, i]
                avg_loss = losses.sum() / num_selected
            else:
                avg_loss = losses.sum() / ((t==i).sum().float())
            loss_list.append(avg_loss)

        return torch.stack(loss_list)

    custom_loss_list = loss_per_class(x, t)
    custom_loss_list_neg = loss_per_class_neg(x, t)
    conf_loss_list = conf_loss_per_class(x, t)
    # P(+)
    risk1 = sum([prior_emp[i]*custom_loss_list[i]/max([1., (t == i).float().sum()]) for i in range(class_num)])

    # P'(-)
    risk2 = sum([prior_emp[i]*conf_loss_list[i] for i in range(class_num)])

    # P(-)
    risk3 = sum([prior_emp[i]*custom_loss_list_neg[i]/max([1., (t == i).float().sum()]) for i in range(class_num)])

    # U'(-)
    risk4 = conf_loss_list[class_num]

    positive_risk = w_p*(risk1 + risk2 - risk3)
    negative_risk = risk4
    if positive_risk < 0:
        positive_risk = torch.tensor(0., requires_grad=True).cuda(ncuda)
    x_out =  positive_risk + negative_risk

    return x_out