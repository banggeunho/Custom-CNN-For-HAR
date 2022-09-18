import torch
import torch.nn as nn

class CNN(torch.nn.Module):
    # [batch, channel, height, width]
    # input[50, 1, 50, 6]
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 1, 50, 6)
        #    Conv     -> (?, 32, 50, 6)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        # L2 ImgIn shape=(?, 32, 50, 6)
        #    Conv      ->(?, 64, 50, 6)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        # L3 ImgIn shape=(?, 64, 50, 6)
        #    Conv      ->(?, 128, 50, 6)
        #    Pool      ->(?, 128, 26, 4)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 6x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(11* 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))

        # L5 Final FC 625 inputs -> 13 outputs
        self.fc2 = torch.nn.Linear(625, 13, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)  # Flatten them for FC
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.fc2(out)
        # print(out.shape)
        return out



class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, input_size, num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_size, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
        	nn.Dropout(),
        	nn.Linear(1792, 128),
        	nn.ReLU(),
        	nn.Dropout(),
        	nn.Linear(128, num_classes),
        	)

    def forward(self, x):
    	x = self.features(x)
    	x = x.view(x.size(0), 1792)
    	out = self.classifier(x)

    	return out
