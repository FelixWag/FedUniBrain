def update_wandb_metrics(wandb_metrics, train_val_test, **kwargs):
    updated_dict = {f"{train_val_test}/{key}": value for key, value in kwargs.items()}
    wandb_metrics.update(updated_dict)

def update_wandb_metrics_client(wandb_metrics, train_val_test, client_name, **kwargs):
    updated_dict = {f"{train_val_test}/{key}_{client_name}": value for key, value in kwargs.items()}
    wandb_metrics.update(updated_dict)
