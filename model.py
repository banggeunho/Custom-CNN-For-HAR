import torch

class CNN(torch.nn.Module):
    # [batch, channel, height, width]
    # input[2, 1, 10, 6]
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 1, 10, 6)
        #    Conv     -> (?, 32, 10, 6)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        # L2 ImgIn shape=(?, 32, 10, 6)
        #    Conv      ->(?, 64, 10, 6)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU())
        # L3 ImgIn shape=(?, 64, 10, 6)
        #    Conv      ->(?, 128, 10, 6)
        #    Pool      ->(?, 128, 6, 4)
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
