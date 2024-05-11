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
import os
import shutil

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

device = torch.device("mps")
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


@torch.no_grad()
def evaluate(net, val_loader):
		net.eval()
		outputs = [net.validation_step(batch) for batch in val_loader]
		return net.validation_epoch_end(outputs)


@torch.no_grad()
def test(net, test_loader):
		net.eval()
		outputs = [net.validation_step(batch) for batch in test_loader]
		return net.validation_epoch_end(outputs)

class BaseNet(nn.Module):
		def training_step(self, data):
				inputs = data[0].to(device)
				labels = data[1].to(device)
				outputs = self(inputs)  # Generate predictions
				loss = F.cross_entropy(outputs, labels)  # Calculate loss
				# loss = F.nll_loss(outputs, labels)
				acc = accuracy(outputs, labels)
				return {"train_loss": loss, "train_acc": acc}

		def validation_step(self, data):
				inputs = data[0].to(device)
				labels = data[1].to(device)
				outputs = self(inputs)  # Generate predictions
				loss = F.cross_entropy(outputs, labels)  # Calculate loss
				# loss = F.nll_loss(outputs, labels)
				acc = accuracy(outputs, labels)  # Calculate accuracy
				return {"val_loss": loss.detach(), "val_acc": acc}

		def validation_epoch_end(self, outputs):
				batch_losses = [x["val_loss"] for x in outputs]
				epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
				batch_accs = [x["val_acc"] for x in outputs]
				epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
				return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

		def epoch_end(self, epoch, result):
				print(
						"Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
								epoch,
								result["train_loss"],
								result["train_acc"],
								result["val_loss"],
								result["val_acc"],
						)
				)

class Net(BaseNet):
		def __init__(self):
				super().__init__()
				self.network = nn.Sequential(
						nn.Conv2d(3, 32, kernel_size=3, padding=1),
						nn.ReLU(),
						nn.Conv2d(32, 64, kernel_size=3, padding=1),
						nn.BatchNorm2d(64),
						nn.ReLU(),
						nn.MaxPool2d(2, 2),
						nn.Conv2d(64, 128, kernel_size=3, padding=1),
						nn.ReLU(),
						nn.Conv2d(128, 128, kernel_size=3, padding=1),
						nn.BatchNorm2d(128),
						nn.ReLU(),
						#nn.MaxPool2d(2, 2),
						nn.Conv2d(128, 256, kernel_size=3, padding=1),
						nn.ReLU(),
						nn.Conv2d(256, 256, kernel_size=3, padding=1),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						#nn.MaxPool2d(2, 2),
						nn.AdaptiveAvgPool2d(1),
						nn.Dropout(0.4),
						nn.Flatten(),
						nn.Linear(256, 1024),
						nn.ReLU(),
						nn.Linear(1024, 512),
						nn.ReLU(),
						nn.Linear(512, num_classes),
				)

		def forward(self, x):
				return self.network(x)


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
						res = net.training_step(data)
						loss = res["train_loss"]
						acc = res["train_acc"]
						train_losses.append(loss)
						train_acc.append(acc)
						loss.backward()
						optimizer.step()
						optimizer.zero_grad()

				result = evaluate(net, val_loader)
				result["train_loss"] = torch.stack(train_losses).mean().item()
				result["train_acc"] = torch.stack(train_acc).mean().item()
				net.epoch_end(epoch, result)
				history.append(result)

		print("Finished Training")
		torch.save(net.state_dict(), PATH)
		return history


def plot_accuracies(history, version):
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
		plt.savefig("versions/{}/acc-graph-{}.png".format(version, version))


def plot_losses(history, version):
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
		plt.savefig("versions/{}/loss-graph-{}.png".format(version, version))


def set_worker_sharing_strategy(worker_id: int) -> None:
		torch.multiprocessing.set_sharing_strategy("file_system")


def set_up_version():
		with open("version.txt", "r+") as f:
				version = f.read()
				version = float(version)
				new_version = round(version + 0.1, 1)
				f.seek(0)
				f.write(str(new_version))
				f.truncate()
				f.close()
				os.mkdir("./versions/{}".format(new_version))
				shutil.copyfile(
						"image-classifier.py",
						"./versions/{}/image-classifier.py".format(new_version),
				)
				return new_version


def test_net(version):
		test_dataset = datasets.ImageFolder(
				test_data_dir,
				transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]),
		)
		test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4)
		net = Net().to(device)
		net.load_state_dict(torch.load(PATH))
		test_result = test(net, test_loader)
		print("Version [{}], loss: {:.4f}, acc: {:.4f}".format(version, test_result["val_loss"], test_result["val_acc"]))


if __name__ == "__main__":
		version = set_up_version()
		train_transform = transforms.Compose(
				[
						transforms.Resize((150, 150)),
						transforms.RandomHorizontalFlip(0.1),
						#transforms.RandomSolarize(.1),
						#transforms.RandomEqualize(.1),
						#transforms.RandomGrayscale(.1),
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
		#val_size = 2000
		#train_size = len(train_dataset) - val_size

		#train_data, val_data = random_split(train_dataset, [train_size, val_size])
		# load the train and validation into batches.
		train_loader = DataLoader(
				train_dataset,
				batch_size,
				shuffle=True,
				num_workers=4,
				pin_memory=True,
				worker_init_fn=set_worker_sharing_strategy,
		)
		val_loader = DataLoader(
				val_data,
				batch_size * 2,
				num_workers=4,
				pin_memory=True,
				worker_init_fn=set_worker_sharing_strategy,
		)

		# to device for gpu
		net = Net().to(device)

		num_epochs = 50
		lr = 0.000001
		# fitting the model on training data and record the result after each epoch
		history = fit(num_epochs, lr, net, train_loader, val_loader)
		plot_accuracies(history, version)
		plot_losses(history, version)
		#test_net(version)
