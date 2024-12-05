#!/usr/bin/env python
# coding: utf-8

import json
import os
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from cust_dataset import CustomDataset
from network import Network
from torch.utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import ShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import argparse
import time
import numpy as np
import pandas as pd
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, explained_variance_score, r2_score
import pdb
from torch.optim.lr_scheduler import StepLR  # Import StepLR scheduler
from scipy.stats import pearsonr


def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    if np.isnan(actual).any():
        print(f"NaN detected in 'actual' values. Number of NaNs: {np.isnan(actual).sum()}")
    else:
        print("No NaN detected in 'actual' values.")
    
    if np.isnan(predicted).any():
        print(f"NaN detected in 'predicted' values. Number of NaNs: {np.isnan(predicted).sum()}")
    else:
        print("No NaN detected in 'predicted' values.")

    r2_vals = r2_score(actual, predicted)
    ev_vals = explained_variance_score(actual, predicted)
    # correlation_test = np.corrcoef(actual,predicted)[0,1]
    correlation, p_value = pearsonr(actual, predicted)

    if np.isnan(correlation):
        print("NaN detected in correlation!")
        print("Actual values during NaN correlation:")
        print(actual)
        print("Predicted values during NaN correlation:")
        print(predicted)
        print("Values after fc1 (before dropout):")
        print(model.fc1_output)  # Values stored in the forward pass
        print("Values after fc2:")
        print(model.fc2_output)  # Values stored in the forward pass


    return {'r2 score':r2_vals, 'ev': ev_vals, 'correlation': correlation, 'p-value': p_value}

# New function to load config
def load_config(config_name):
    with open('config.json') as config_file:
        configs = json.load(config_file)
    return configs[config_name]

# Argument parser for dynamic config selection
parser = argparse.ArgumentParser(description='Cross-validation with dynamic config')
parser.add_argument('--config', type=str, required=True, help='Config name from the config.json file (e.g., config1)')
args = parser.parse_args()

# Load the selected configuration
config = load_config(args.config)
learning_rate_layer = config['learning_rate_layer']
dropout_value = config['dropout']
fc1_input_size = config['fc1_input_size']
fc1_output_size = config['fc1_output_size']
fc2_output_size = config['fc2_output_size']

print(f"Using config: {args.config}")
print(f"Learning Rate Layer: {learning_rate_layer}")
print(f"Dropout: {dropout_value}")
print(f"fc1_input_size: {fc1_input_size}, fc1_output_size: {fc1_output_size}, fc2_output_size: {fc2_output_size}")

#setup initialization
torch.manual_seed(52)
num_workers = 4
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clean old checkpoints
os.system('rm -rf checkpoint_fold_*.pth')

#data initialization
data = CustomDataset(transform =
                        transforms.Compose([
                            transforms.RandomHorizontalFlip()
                            ]))

# Initialize lists to store metrics for all folds
best_train_mse_list, best_train_r2_list, best_train_correlation_list, best_train_ev_list = [], [], [], []
best_valid_mse_list, best_valid_r2_list, best_valid_correlation_list, best_valid_ev_list = [], [], [], []

# prepare for k-fold
kf = KFold(n_splits=4, shuffle=True, random_state=52)

unique_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
print(f"Unique ID for this job: {unique_id}")

#Early Stopping
class EarlyStopping:
        def __init__(self, patience=10, verbose=False, delta=0, fold=None, unique_id=None):
      
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')
            self.delta = delta
            self.fold = fold
            self.unique_id = unique_id
            self.checkpoint_filename = f'checkpoint_{self.unique_id}_fold_{self.fold}.pth'
    
            # Log unique id
            print(f"Unique ID for this job: {self.unique_id}")

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
           
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model for fold {self.fold}...')
                torch.save(model.state_dict(), self.checkpoint_filename)
                # torch.save(model.state_dict(), checkpoint_filename)  # Save model per fold with unique file name

                self.val_loss_min = val_loss

scaler = GradScaler()


fold_valid_losses = []
fold_checkpoint_filenames = []
fold_numbers = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(data.train_idx)):
   
    print(f"Training fold {fold}...")
    best_valid_loss = float('inf')

  

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # test_sampler = SubsetRandomSampler(data.test_idx)


    train_loader = DataLoader(data,batch_size=batch_size,
                                sampler= train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(data,batch_size=batch_size,
                                sampler= valid_sampler, num_workers=num_workers)
    # test_loader = DataLoader(data,batch_size=batch_size,
                                # sampler= test_sampler, num_workers=num_workers)
    # model = Network()
    # model = Network(dropout=dropout_value)  # Pass the loaded dropout value
    model = Network(dropout=dropout_value, fc1_input_size=fc1_input_size, fc1_output_size=fc1_output_size, fc2_output_size=fc2_output_size)


    model.to(device)
    print(model)
    early_stopping = EarlyStopping(patience=20, verbose=True, fold=fold, unique_id=unique_id)
    epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate_layer)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=10, min_lr=1e-6,cooldown=1)

    #training
    print('Starting to Train...')
    best_loss = float('inf')

    for e in range(1,epochs+1):
        model.train()
        train_loss = 0
        actual_batch_values_train = []
        predicted_batch_values_train  = []

        # with torch.no_grad():
        for X, y in train_loader:
            optimizer.zero_grad()
            
            X, y = X.to(device).float(), y.to(device).float()
            X = torch.unsqueeze(X, 1).float()
           
            pred = model(X)
            pred = pred.squeeze()
            loss = criterion(pred, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += (loss.item()*X.shape[0])

            if pred.dim() > 0:
                actual_batch_values_train.extend(y.cpu().numpy())
                predicted_batch_values_train.extend(pred.cpu().detach().numpy())
        
        avg_train_loss = train_loss / len(train_idx)
        train_metrics = calculate_metrics(actual_batch_values_train, predicted_batch_values_train)
        
    
        
        print(f"Epoch {e}/{epochs} - Average train loss: {avg_train_loss:.4f} - "
      f"R2: {train_metrics['r2 score']:.4f}, EV: {train_metrics['ev']:.4f}, "
      f"Correlation: {train_metrics['correlation']:.4f}, P-value: {train_metrics['p-value']:.4e}")
           
        #validation
        
        model.eval()
        valid_loss = 0
        actual_batch_values_valid = []
        predicted_batch_values_valid = []

        with torch.no_grad():
            # with autocast(device_type=device.type):

                for X,y in valid_loader:
                    # Check for NaN in inputs and targets
                    assert not torch.isnan(X).any(), "NaN detected in validation input data"
                    assert not torch.isnan(y).any(), "NaN detected in validation target data"
                    X, y = X.to(device).float(), y.to(device).float()
                    X = torch.unsqueeze(X, 1).float()

                    pred = model(X)
                    pred = pred.squeeze()
                    loss = criterion(pred, y)
                    valid_loss += ((loss.item())*X.shape[0])

                    if pred.dim() > 0:
                        actual_batch_values_valid.extend(y.cpu().numpy())
                        predicted_batch_values_valid.extend(pred.cpu().detach().numpy())

        avg_valid_loss = valid_loss / len(valid_idx)
    
        validation_metrics = calculate_metrics(actual_batch_values_valid, predicted_batch_values_valid)
        # print(f"Epoch {e}/{epochs} - Average validation loss: {avg_valid_loss} - R2: {validation_metrics['r2 score']:.4f}, EV: {validation_metrics['ev']:.4f}, Correlation: {validation_metrics['correlation']:.4f}")

        print(f"Epoch {e}/{epochs} - Average validation loss: {avg_valid_loss:.4f} - "
      f"R2: {validation_metrics['r2 score']:.4f}, EV: {validation_metrics['ev']:.4f}, "
      f"Correlation: {validation_metrics['correlation']:.4f}, P-value: {validation_metrics['p-value']:.4e}")
       
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        scheduler.step(avg_valid_loss)  # Monitor validation loss


        if avg_valid_loss < best_valid_loss:
            best_train_loss = avg_train_loss
            best_train_r2 = train_metrics['r2 score']
            best_train_correlation = train_metrics['correlation']
            best_train_ev = train_metrics['ev']

            best_valid_loss = avg_valid_loss
            best_valid_r2 = validation_metrics['r2 score']
            best_valid_correlation = validation_metrics['correlation']
            best_valid_ev = validation_metrics['ev']


    best_train_mse_list.append(best_train_loss)
    best_train_r2_list.append(best_train_r2)
    best_train_correlation_list.append(best_train_correlation)
    best_train_ev_list.append(best_train_ev)

    best_valid_mse_list.append(best_valid_loss)
    best_valid_r2_list.append(best_valid_r2)
    best_valid_correlation_list.append(best_valid_correlation)
    best_valid_ev_list.append(best_valid_ev)



    checkpoint_filename = early_stopping.checkpoint_filename
    # Store validation losses and checkpoint filenames
    fold_valid_losses.append(best_valid_loss)
    fold_checkpoint_filenames.append(checkpoint_filename)
    fold_numbers.append(fold)
    print(f"Fold {fold} completed.")


# After all folds, find the best fold
best_fold_index = np.argmin(fold_valid_losses)
best_fold = fold_numbers[best_fold_index]
best_checkpoint_filename = fold_checkpoint_filenames[best_fold_index]
print(f"Best fold is {best_fold} with validation loss {fold_valid_losses[best_fold_index]}")

    # Initialize the model and load the best model
model = Network(dropout=dropout_value, fc1_input_size=fc1_input_size,
                fc1_output_size=fc1_output_size, fc2_output_size=fc2_output_size)
model.to(device)
model.load_state_dict(torch.load(best_checkpoint_filename), strict=False)
print(f"Loaded best model from fold {best_fold} checkpoint {best_checkpoint_filename}")

# Define test_sampler and test_loader outside the fold loop
test_sampler = SubsetRandomSampler(data.test_idx)
test_loader = DataLoader(data, batch_size=batch_size,
                         sampler=test_sampler, num_workers=num_workers)
   
#testing 
test_loss = 0
actual_batch_values_test = []
predicted_batch_values_test = []
gradients_list = []  # Store gradients for each sample in the test set
input_gradients_list = []  


model.eval()
criterion = nn.MSELoss()

    # with torch.no_grad():
        # with autocast(device_type='cuda', dtype=torch.float16):
        # with autocast(device_type=device.type):
for X, y in test_loader:
        X, y = X.to(device).float(), y.to(device).float()
        X = torch.unsqueeze(X, 1).float()
        X.requires_grad = True  # Enable gradient computation for interpretability


        pred = model(X)
        pred = pred.squeeze()
        loss = criterion(pred, y)
        test_loss += (loss.item() * X.shape[0])
        if pred.dim() > 0:
            actual_batch_values_test.extend(y.cpu().numpy())
            predicted_batch_values_test.extend(pred.cpu().detach().numpy())
            model.zero_grad()  # Zero out gradients for each sample
            pred.sum().backward()  # Compute gradients
            gradients = X.grad.detach()  # Store gradients as numpy array
            input_gradients = X.detach() * gradients
            input_gradients_np = input_gradients.cpu().numpy()
            gradients_np = gradients.cpu().numpy()


            gradients_list.append(gradients_np)  # Append gradients to the list
            input_gradients_list.append(input_gradients_np)

avg_test_loss = test_loss / len(data.test_idx)
test_metrics = calculate_metrics(actual_batch_values_test, predicted_batch_values_test)
# print(f"Average test loss: {avg_test_loss}")
# print(f"Test - Average test loss: {avg_test_loss:.4f} - R2: {test_metrics['r2 score']:.4f}, "
#       f"EV: {test_metrics['ev']:.4f}, Correlation: {test_metrics['correlation']:.4f}")

print(f"Test - Average test loss: {avg_test_loss:.4f} - "
      f"R2: {test_metrics['r2 score']:.4f}, EV: {test_metrics['ev']:.4f}, "
      f"Correlation: {test_metrics['correlation']:.4f}, P-value: {test_metrics['p-value']:.4e}")


# Convert gradients list to a numpy array and save
gradients_array = np.concatenate(gradients_list, axis=0)
np.save(f"all_gradients_best_fold_{best_fold}.npy", gradients_array)
print(f"Saved gradients for best fold {best_fold} to all_gradients_best_fold_{best_fold}.npy")

# Concatenate and save input-gradient products
input_gradients_array = np.concatenate(input_gradients_list, axis=0)
np.save(f"input_gradients_best_fold_{best_fold}.npy", input_gradients_array)
print(f"Saved input-gradient products for best fold {best_fold} to input_gradients_best_fold_{best_fold}.npy")

    
print(f"Testing for best fold {best_fold} completed.")


# Calculate mean and standard deviation for the best training and validation metrics
mean_best_train_mse, std_best_train_mse = np.mean(best_train_mse_list), np.std(best_train_mse_list)
mean_best_train_r2, std_best_train_r2 = np.mean(best_train_r2_list), np.std(best_train_r2_list)
mean_best_train_correlation, std_best_train_correlation = np.mean(best_train_correlation_list), np.std(best_train_correlation_list)
mean_best_train_ev, std_best_train_ev = np.mean(best_train_ev_list), np.std(best_train_ev_list)

mean_best_valid_mse, std_best_valid_mse = np.mean(best_valid_mse_list), np.std(best_valid_mse_list)
mean_best_valid_r2, std_best_valid_r2 = np.mean(best_valid_r2_list), np.std(best_valid_r2_list)
mean_best_valid_correlation, std_best_valid_correlation = np.mean(best_valid_correlation_list), np.std(best_valid_correlation_list)
mean_best_valid_ev, std_best_valid_ev = np.mean(best_valid_ev_list), np.std(best_valid_ev_list)

# # Calculate mean and standard deviation for test metrics
# mean_mse, std_mse = np.mean(mse_list), np.std(mse_list)
# mean_r2, std_r2 = np.mean(r2_list), np.std(r2_list)
# mean_correlation, std_correlation = np.mean(correlation_list), np.std(correlation_list)
# mean_ev, std_ev = np.mean(ev_list), np.std(ev_list)

# Print the results
print(f"Final results for job with unique_id: {early_stopping.unique_id}")

print(f"Mean Best Training MSE: {mean_best_train_mse:.4f} +/- {std_best_train_mse:.4f}")
print(f"Mean Best Training R2: {mean_best_train_r2:.4f} +/- {std_best_train_r2:.4f}")
print(f"Mean Best Training Pearson Correlation: {mean_best_train_correlation:.4f} +/- {std_best_train_correlation:.4f}")
print(f"Mean Best Training Explained Variance: {mean_best_train_ev:.4f} +/- {std_best_train_ev:.4f}")

print(f"Mean Best Validation MSE: {mean_best_valid_mse:.4f} +/- {std_best_valid_mse:.4f}")
print(f"Mean Best Validation R2: {mean_best_valid_r2:.4f} +/- {std_best_valid_r2:.4f}")
print(f"Mean Best Validation Pearson Correlation: {mean_best_valid_correlation:.4f} +/- {std_best_valid_correlation:.4f}")
print(f"Mean Best Validation Explained Variance: {mean_best_valid_ev:.4f} +/- {std_best_valid_ev:.4f}")

# Directly print test metrics
print(f"Test MSE: {avg_test_loss:.4f}")
print(f"Test R2: {test_metrics['r2 score']:.4f}")
print(f"Test Pearson Correlation: {test_metrics['correlation']:.4f}")
print(f"Test Explained Variance: {test_metrics['ev']:.4f}")
# print(f"Mean Test MSE: {mean_mse:.4f} +/- {std_mse:.4f}")
# print(f"Mean Test R2: {mean_r2:.4f} +/- {std_r2:.4f}")
# print(f"Mean Test Pearson Correlation: {mean_correlation:.4f} +/- {std_correlation:.4f}")
# print(f"Mean Test Explained Variance: {mean_ev:.4f} +/- {std_ev:.4f}")

print('####################################################################')
print("Done")
print("End")
