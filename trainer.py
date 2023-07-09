import numpy as np
from torch import Tensor
import typing as t
from dataloader import DataLoader
from loss import Loss, MSELoss
from nn import NeuralNetwork
from optimizer import Optimizer
import logging
import matplotlib.pyplot as plt


DEFAULT_EPOCHS = 5000
logger = logging.getLogger("trainer")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

file_handler = logging.FileHandler("log.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train(
    nn: NeuralNetwork,
    data_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int = DEFAULT_EPOCHS,
    loss: Loss = MSELoss(),
    validator_data_loader: t.Optional[DataLoader] = None,
) -> None:
    full_learning_loss = []
    epoch_loss_array = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_id = 0
        for batch in data_loader:
            predicted = nn.forward(batch.inputs)
            this_loss = loss.loss(predicted, batch.targets)
            full_learning_loss.append(this_loss)
            epoch_loss += this_loss
            grad = loss.grad(predicted, batch.targets)
            nn.zero_grads()
            nn.backward(grad)
            optimizer.step()
            logger.debug(f"batch {batch_id} completed: {this_loss:%.2f} loss")
            batch_id += 1
        logger.info(f"epoch: {epoch}: {epoch_loss:%.2f} loss")
        epoch_loss_array.append(epoch_loss)

    _, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].plot(range(len(full_learning_loss)), full_learning_loss, "full training loss")
    axs[0, 0].set_title("Training graph")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(range(len(epoch_loss_array)), epoch_loss_array, "epoch loss")
    axs[0, 1].set_title("Training graph")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()

    if validator_data_loader is not None:
        logger.info("Has validator set, running...")
        validator_loss_array: t.List[float] = []
        epoch_loss = 0.0
        batch_id = 0
        for batch in validator_data_loader:
            predicted = nn.forward(batch.inputs)
            this_loss = loss.loss(predicted, batch.targets)
            epoch_loss += this_loss
            logger.debug(f"validator batch {batch_id} completed: {this_loss:%.2f} loss")
            batch_id += 1
        logger.info(f"validator epoch: {epoch}: {epoch_loss:%.2f} loss")
        validator_loss_array.append(epoch_loss)
        axs[1, 0].plot(range(len(validator_loss_array)), validator_loss_array, "validator loss")
        axs[1, 0].set_title("Validation graph")
        axs[1, 0].set_xlabel("Epochs")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].legend()  

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()