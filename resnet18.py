import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

PATH = "./image-classifier-cnn.pth"

# train and test data directory
### Seg ###

data_dir = "./input/seg/train"
test_data_dir = "./input/seg/test"
num_classes = 6
"""
### Blood Cells ###
data_dir = "./input/blood-cell/train"
test_data_dir = "./input/blood-cell/test"
num_classes = 4
"""


def accuracy(outputs, labels):
	_, preds = torch.max(outputs, dim=1)
	return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def validation_step(net, data):
	inputs = data[0]
	labels = data[1]
	outputs = net(inputs)  # Generate predictions
	loss = F.cross_entropy(outputs, labels)  # Calculate loss
	acc = accuracy(outputs, labels)  # Calculate accuracy
	return {"val_loss": loss.detach(), "val_acc": acc}

def validation_epoch_end(outputs):
	batch_losses = [x["val_loss"] for x in outputs]
	epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
	batch_accs = [x["val_acc"] for x in outputs]
	epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
	return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

@torch.no_grad()
def evaluate(net, val_loader):
	net.eval()
	outputs = [validation_step(net, batch) for batch in val_loader]
	return validation_epoch_end(outputs)

def fit(epochs, lr, net, train_loader, val_loader):
	history = []
	optimizer = optim.Adam(net.parameters(), lr=lr)

	for epoch in range(epochs):  # loop over the dataset multiple times
		train_losses = []
		train_acc = []
		net.train()
		loop = tqdm(train_loader, unit="it")
		loop.set_description(f"Epoch {epoch}")
		for i, data in enumerate(loop):
			inputs = data[0]
			labels = data[1]
			outputs = net(inputs)  # Generate predictions
			loss = F.nll_loss(outputs, labels)  # Calculate loss
			loss.backward()
			acc = accuracy(outputs, labels)
			train_losses.append(loss)
			train_acc.append(acc)
			optimizer.step()
			optimizer.zero_grad()

		result = evaluate(net, val_loader)
		result["train_loss"] = torch.stack(train_losses).mean().item()
		result["train_acc"] = torch.stack(train_acc).mean().item()
		print(
			"Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
				epoch,
				result["train_loss"],
				result["train_acc"],
				result["val_loss"],
				result["val_acc"],
			)
		)
		history.append(result)

	print("Finished Training")
	return history


def plot_accuracies(history):
	"""Plot the history of accuracies"""
	plt.clf()
	train_accuracies = [x["train_acc"] for x in history]
	val_accuracies = [x["val_acc"] for x in history]
	plt.plot(train_accuracies, "-bx")
	plt.plot(val_accuracies, "-rx")
	plt.xlabel("epoch")
	plt.ylabel("accuracy")
	plt.legend(["Training", "Validation"])
	plt.title("Accuracy vs. No. of epochs")
	plt.savefig("acc-graph-resnet18.png")


def plot_losses(history):
	"""Plot the losses in each epoch"""
	plt.clf()
	train_losses = [x.get("train_loss") for x in history]
	val_losses = [x["val_loss"] for x in history]
	plt.plot(train_losses, "-bx")
	plt.plot(val_losses, "-rx")
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.legend(["Training", "Validation"])
	plt.title("Loss vs. No. of epochs")
	plt.savefig("loss-graph-resnet18.png")

if __name__ == "__main__":
	train_transform = transforms.Compose(
		[
			transforms.Resize((150, 150)),
			transforms.RandomHorizontalFlip(0.1),
			# transforms.RandomSolarize(.1),
			# transforms.RandomEqualize(.1),
			# transforms.RandomGrayscale(.1),
			transforms.ToTensor(),
		]
	)
	# load the train and test data
	train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

	val_data = datasets.ImageFolder(
		test_data_dir,
		transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]),
	)

	batch_size = 128

	train_loader = DataLoader(
		train_dataset,
		batch_size,
		shuffle=True,
		num_workers=4,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_data,
		batch_size * 2,
		num_workers=4,
		pin_memory=True,
	)

	# to device for gpu
	net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, num_classes)

	num_epochs = 30
	lr = 0.001
	# fitting the model on training data and record the result after each epoch
	history = fit(num_epochs, lr, net, train_loader, val_loader)
	plot_accuracies(history)
	plot_losses(history)
