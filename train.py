import torch.autograd
import torch.nn as nn
from dataset import Dataset2


class Network2(nn.Module):
    def __init__(self, input_dim=20, output_dim=7):
        super(Network2, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.dis(x)
        return x


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()
    return correct / total


if __name__ == '__main__':
    num_epoch = 500
    data_vector_dim = 20
    item_of_single_class = 10
    train_dataset = Dataset2(item_of_single_class=item_of_single_class, data_vector_dim=data_vector_dim)
    test_dataset = Dataset2(item_of_single_class=item_of_single_class, data_vector_dim=data_vector_dim)
    class_num = len(train_dataset.noise_scope_list)
    dataset_length = item_of_single_class * class_num

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=dataset_length, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=dataset_length, shuffle=True
    )

    net = Network2(input_dim=data_vector_dim, output_dim=class_num)
    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(num_epoch):
        for i, (data_batch, label_batch) in enumerate(train_dataloader):
            data_batch = data_batch.cuda()
            real_out = net(data_batch)
            label_batch = label_batch.cuda()
            loss = criterion(real_out, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("acc:{:.6f}".format(evaluate(net, test_dataloader)))
