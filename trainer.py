import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, device, num_epochs, checkpoint_dir, 
                 warmup_steps, max_steps, min_lr=1e-6, log_interval=100):
        super().__init__()  # Initialize Subject
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval

        self.scheduler = self._get_lr_scheduler(warmup_steps, max_steps, min_lr)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.train_steps = []
        self.val_steps = []

    def _get_lr_scheduler(self, warmup_steps, max_steps, min_lr):
        warmup_scheduler = lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=lambda step: step / warmup_steps if step < warmup_steps else 1.0
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max_steps - warmup_steps, 
            eta_min=min_lr
        )
        return lr_scheduler.SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_steps]
        )

    def _evaluate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in self.valid_loader:
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)
                _, val_batch_loss = self.model(val_inputs, val_targets)
                val_loss += val_batch_loss.item()
        
        avg_val_loss = val_loss / len(self.valid_loader)
        self.val_losses.append(avg_val_loss)
        self.val_steps.append(self.global_step)

        # Notify observers about validation loss
        # Notify observers and check stopping condition
        should_stop = self.notify_observers("validation", {"val_loss": avg_val_loss})

        return avg_val_loss, should_stop
    
    def _save_checkpoint(self, epoch, loss, filename):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }, os.path.join(self.checkpoint_dir, filename))

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            batch_loss = 0
            start_time = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                logits, loss = self.model(inputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                current_loss = loss.item()
                batch_loss += current_loss
                epoch_loss += current_loss
                self.global_step += 1

                 # Notify observers at batch end
                self.notify_observers("batch_end", {
                    "batch_idx": batch_idx, 
                    "total_batches": len(self.train_loader),
                    "loss": current_loss, 
                    "lr": self.scheduler.get_last_lr()[0]
                })
                
                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    avg_loss = batch_loss / self.log_interval
                    elapsed = time.time() - start_time
                    print(f"| epoch {epoch+1:3d} | {batch_idx:5d}/{len(self.train_loader):5d} batches | "
                          f"lr {self.scheduler.get_last_lr()[0]:.6f} | ms/batch {elapsed * 1000 / self.log_interval:5.2f} | "
                          f"loss {avg_loss:5.2f}")
                    self.train_losses.append(avg_loss)
                    self.train_steps.append(self.global_step)
                    batch_loss = 0
                    start_time = time.time()
                
                if self.global_step % (10 * self.log_interval) == 0:
                    avg_val_loss, should_stop = self._evaluate()
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self._save_checkpoint(epoch, self.best_val_loss, 'best_model.pt')
                        print(f"| Saved best model to {os.path.join(self.checkpoint_dir, 'best_model.pt')}")
                    
                    if should_stop:
                        print("Training stopped early due to early stopping.")
                        return

                    self.model.train()
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"| End of epoch {epoch+1} | Average loss: {avg_epoch_loss:5.2f}")
            self._save_checkpoint(epoch, avg_epoch_loss, f'checkpoint_epoch_{epoch+1}.pt')
        
        self._plot_loss()
        return self.train_losses, self.val_losses, self.train_steps, self.val_steps

    def _plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_steps, self.train_losses, label='Training Loss', color='blue', linestyle='-', marker='o')
        plt.plot(self.val_steps, self.val_losses, label='Validation Loss', color='red', linestyle='-', marker='x')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.checkpoint_dir, "loss_curve.png"))