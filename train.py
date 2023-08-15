import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import evals


class VCNTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        epochs: int,
        device,
        checkpoint: str,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.checkpoint = checkpoint

    def train(self, verbose=True):
        train_losses = []
        train_pixel_accuracies = []
        val_losses = []
        val_pixel_accuracies = []

        optimizer = self.optimizer

        for epoch in range(self.epochs):
            # set mode to train
            desc = f"Epoch [{epoch+1}/{self.epochs}]"
            pbar = tqdm(self.train_loader, leave=False, desc=desc)
            total_loss = 0

            for i, (X, Y) in enumerate(pbar):
                self.model.train()

                X = X.to(self.device)
                Y = Y.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, Y)
                total_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss evaluations
                train_losses.append(loss.item())
                val_loss = evals.loss(self.model, self.criterion, self.val_loader)
                val_losses.append(val_loss)

                batch_summary = f"Loss: {loss.item()}, Val Loss: {val_loss}"
                pbar.set_postfix(summary=batch_summary)

            # evaluations (pixel accuracy)
            train_pixel_acc = evals.pixel_accuracy(self.model, self.train_loader)
            val_pixel_acc = evals.pixel_accuracy(self.model, self.val_loader)

            train_pixel_accuracies.append(train_pixel_acc)
            val_pixel_accuracies.append(val_pixel_acc)
            
            train_loss = total_loss / len(self.train_loader)
            epoch_summary = f"Epoch [{epoch+1}/{self.epochs}]: Loss: {train_loss}, Val Loss: {val_loss} Train Pixel Acc: {train_pixel_acc}, Val Pixel Acc: {val_pixel_acc}"

            if verbose:
                print(epoch_summary)
                

        iterations = len(train_losses)

        checkpoint_data = {
            "state_dict": self.model.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_pixel_accuracies": train_pixel_accuracies,
            "val_pixel_accuracies": val_pixel_accuracies,
            "iterations": iterations,
            "epochs": self.epochs
        }

        # save checkpoint
        torch.save(checkpoint_data, self.checkpoint)
        
        # empty gpu so that we can train other models with out of memory error
        torch.cuda.empty_cache()
