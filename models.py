import torch


class MobileRNN(torch.nn.Module):
    def __init__(self, weights, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.mn = load_model(weights, output_size=output_size)
        self.rnn = torch.nn.RNN(input_size=output_size, \
            hidden_size=hidden_size, num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_size, 7)
    
    def forward(self, x):
        # images = [x[:, 3*i:3*(i+1), :, :] for i in range(int(x.shape[1]/3))]
        bs, nc, w, h = x.shape
        seq_len = int(nc/3)
        images = torch.Tensor.view(bs*seq_len,3,w,h)
        # mn_images = [self.mn(im).unsqueeze(0) for im in images]
        mn_images_concatenated = self.mn(images).view(seq_len, bs, 3, -1)
        # first_image = x[:, :3, :, :] #batch_size, num_channels (first 3), width, height
        # second_image = x[:, 3:, :, :] #batch_size, num_channels (last 3), width, height
        # mn_first_image = self.mn(first_image).unsqueeze(0)
        # mn_second_image = self.mn(second_image).unsqueeze(0)
        # mn_image_first_second = torch.cat((mn_first_image,mn_second_image), dim=0)
        # mn_images_concatenated= torch.cat(mn_images, dim=0)
        out, hn = self.rnn(mn_images_concatenated)
        pose = self.linear(out[-1]) # take output of last time step
        return pose




def load_model(weights_path=None, device = 'cuda:0', output_size=7):
    """
    Loads MobileNetV2 pre-trained on ImageNet from PyTorch's cloud.
    Modifies last layers to fit our pose regression problem.
    """
    # Base model is MobileNetV2 from PyTorch's hub
    model = torch.hub.load(
        'pytorch/vision:v0.9.0',
        'mobilenet_v2',
        pretrained=True
    )

    # We modify the classifier of MobileNetV2 with a custom regressor
    in_features = list(model.classifier.children())[-1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=in_features,
            out_features=2048,
            bias=True
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            in_features=2048,
            out_features=output_size,
            bias=True
        )
    )

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device(device))['model_state_dict'])
    return model
