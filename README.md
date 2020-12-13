# use-of-tensorboardx
Some usages of tensorboardx.
The code can be run in the environment below:
```requirements.txt
python==3.8
pytorch==1.6
tensorboardX==2.1
```
After run the `train.py`, a `runs` directory is auto generated.
The run the command below.
```bash
tensorboard --logdir runs
```
Copy the url the command result suggest you to copy, then paste the url in your browser. You can see the loss curve, acc curve and the network graph.
