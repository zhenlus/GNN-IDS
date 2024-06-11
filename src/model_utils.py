import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

def train(model, lr, num_epochs, X_train, Y_train, X_val, Y_val, edge_index, rt_meas_dim=78, device='cpu'):

    # weighted cross entropy loss
    num_class_0 = (Y_train == 0).sum().item()
    num_class_1 = (Y_train == 1).sum().item()
    pos_weight = torch.tensor([num_class_0 / num_class_1], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    action_mask = model.action_mask
    if model.name == 'NN':
        X_train = X_train[:, action_mask, -rt_meas_dim:].clone()
        X_val = X_val[:, action_mask, -rt_meas_dim:].clone()
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=30, shuffle=False)

    stat = {}
    stat['loss_train'] = []
    stat['loss_val'] = [] 
    stat['acc_train'] = []
    stat['acc_val'] = []

    for epoch in range(num_epochs):
        model.train()
        # start mini-batch training
        for batch_X, batch_y in dataloader:
            if model.name == 'NN':
                output = model(batch_X)
                loss = criterion(output, batch_y)
            elif model.name == 'GAT':
                loss = 0
                for i in range(len(batch_X)):
                    output = model(batch_X[i], edge_index)
                    loss += criterion(output[action_mask], batch_y[i])
                loss /= len(batch_X)
            else:
                output = model(batch_X, edge_index)
                loss = criterion(output[:, action_mask], batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_train, acc_train = evaluate_loss_acc(model, X_train, Y_train, criterion, edge_index, device)
        loss_val, acc_val     = evaluate_loss_acc(model, X_val, Y_val, criterion, edge_index, device=device)

        stat['loss_train'].append(loss_train)
        stat['loss_val'].append(loss_val)
        stat['acc_train'].append(acc_train)
        stat['acc_val'].append(acc_val)
        
        # early stop if overfitting is observed on the validation set
        if (epoch + 1) % 10 == 0:
            print('Epoch: {:03d}, Training Loss: {:.4f}, Traning Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch + 1, loss_train, acc_train, loss_val, acc_val))
        
    model.stat = stat

def evaluate_loss_acc(model, X, y, criterion, edge_index, device='cpu'):
    model.eval()
    mask = model.action_mask
    loss, acc = 0, 0
    with torch.no_grad():
        if model.name in ['NN']:
            output = model(X)
            loss = criterion(output, y)
            y_pred = torch.sigmoid(output) > 0.5
            acc = (y_pred == y).sum().item() / (y.shape[0] * y.shape[1])
        elif model.name == 'GAT':
            true_pred = []
            for i in range(len(X)):
                output = model(X[i], edge_index)
                loss += criterion(output[mask], y[i])
                y_pred = torch.sigmoid(output[mask]) > 0.5
                true_pred.append(y_pred)
            loss /= len(X)
            acc = (torch.stack(true_pred, dim=0) == y).sum().item() / (y.shape[0] * y.shape[1])
        else:
            output = model(X, edge_index)
            loss = criterion(output[:, mask], y)
            y_pred = torch.sigmoid(output[:, mask]) > 0.5
            acc = (y_pred == y).sum().item() / (y.shape[0] * y.shape[1])
    return loss, acc


def predict_prob(model, X, edge_index, rt_meas_dim=78, device='cpu'):
    model.eval()
    mask = model.action_mask
    prob = torch.zeros((len(X), len(mask), 2), dtype=torch.float32, device=device)
    with torch.no_grad():
        if model.name == 'NN':
            prob_1 = torch.sigmoid(model(X[:,:,-rt_meas_dim:]))[:, mask]
            prob = torch.stack([1 - prob_1, prob_1], dim=2)
        elif model.name == 'GAT':
            prob_1 = [torch.sigmoid(model(X[i], edge_index))[mask] for i in range(len(X))]
            prob = torch.stack([1 - torch.stack(prob_1, dim=0), torch.stack(prob_1, dim=0)], dim=2)
        else:
            prob_1 = torch.sigmoid(model(X, edge_index))[:, mask]
            prob = torch.stack([1 - prob_1, prob_1], dim=2)
    return prob

def evaluate_acc(model, X, y, edge_index, device='cpu'):
    prob = predict_prob(model, X, edge_index)
    pred = torch.argmax(prob, dim=2)
    accuracy = (pred == y).sum().item() / (y.shape[0] * y.shape[1])
    return accuracy

def evaluate_performance(models, X, y, edge_index, device='cpu'):
    metrics = []
    for name, model in models.items():
        model.eval()  
        prob = predict_prob(model, X, edge_index)
        pred_ts = torch.argmax(prob, dim=2)
        accuracy = (pred_ts == y).sum().item() / (y.shape[0] * y.shape[1])
        conf_matrix = confusion_matrix(y.flatten(), pred_ts.flatten())
        precision, recall, f1, _ = precision_recall_fscore_support(y.view(-1), pred_ts.view(-1), average='macro')

        TN, FP, FN, TP = conf_matrix.ravel()
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        
        y_probs = prob.view(-1, 2)
        fpr, tpr, thresholds = roc_curve(y.view(-1), y_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        d = {'model': name, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
                'precision': '{:.4f}'.format(precision), 'recall': '{:.4f}'.format(recall), 'f1': '{:.4f}'.format(f1),
                'auc': '{:.4f}'.format(roc_auc), 'fpr': '{:.4f}'.format(FPR), 'fnr': '{:.4f}'.format(FNR),
                'loss_train': '{:.4f}'.format(model.stat['loss_train'][-1]), 'loss_val': '{:.4f}'.format(model.stat['loss_val'][-1]),
                'acc_train': '{:.4f}'.format(model.stat['acc_train'][-1]), 'acc_val': '{:.4f}'.format(model.stat['acc_val'][-1]),
                'accuracy': '{:.4f}'.format(accuracy)
            }
        metrics.append(d)

    return metrics