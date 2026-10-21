import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    
    This is the standard PyTorch training pattern you'll use everywhere.
    Now you can use torch.optim to handle the gradient updates!
    
    Args:
        model: nn.Module to train
        X_train: training features, shape (N, ...)
        y_train: training labels, shape (N,)
        X_val: validation features, shape (M, ...)
        y_val: validation labels, shape (M,)
        epochs: number of training epochs
        batch_size: mini-batch size
        lr: learning rate
    
    Returns:
        history: List of dicts, one per epoch, with keys:
            - 'epoch': epoch number (starting from 1)
            - 'train_loss': average training loss for the epoch
            - 'val_loss': validation loss after the epoch
            - 'val_accuracy': validation accuracy after the epoch
    
    Steps:
        1. Create optimizer: optim.Adam(model.parameters(), lr=lr)
        2. Create loss function: nn.CrossEntropyLoss()
        3. For each epoch:
            a. Shuffle training data
            b. Loop over mini-batches:
                - optimizer.zero_grad()
                - Forward pass
                - Compute loss
                - loss.backward()
                - optimizer.step()
            c. Compute validation accuracy
            d. Append metrics to history
        4. Return history
    
    Hints:
        - torch.randperm(n) gives a random permutation for shuffling
        - Use model.train() before training, model.eval() before validation
        - Use torch.no_grad() during validation
        - logits.argmax(dim=1) gives predicted classes
    """
    # TODO: Implement the training loop
    n = y_train.shape
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    history = []
    
    # Your code here
    for epoch in range(1, epochs+1):
        # Shuffle training data
        perm = torch.randperm(X_train.size(0))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        model.train()
        train_loss = 0.0
        
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss_value = loss(logits, y_batch)
            loss_value.backward()
            optimizer.step()
            
            train_loss += loss_value.item() * X_batch.size(0)
        
        train_loss /= X_train.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            logits_val = model(X_val)
            val_loss_value = loss(logits_val, y_val)
            val_loss += val_loss_value.item() * X_val.size(0)
            predictions = logits_val.argmax(dim=1)
            correct += (predictions == y_val).sum().item()
        
        val_loss /= X_val.size(0)
        val_accuracy = correct / X_val.size(0)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
    
    return history
