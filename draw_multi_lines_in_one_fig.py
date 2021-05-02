from tensorboardX import SummaryWriter

if __name__ == '__main__':
    with SummaryWriter('./runs/scalar/train_loss') as writer:
        for i in range(5):
            writer.add_scalar('loss', i * 2, i)
    with SummaryWriter('./runs/scalar/test_loss') as writer:
        for i in range(5):
            writer.add_scalar('loss', i * 3, i)
