# STSRNet: Self-texture Transfer Super-resolution and Refocusing Network
The implementation of STSRNet.
## Model
Model and toolkits can be found at lib directory. We recommend Pycharm to avoid dependency problems.
![network architecture](./assets/images/network.jpg)
## Train
see train.py and for more details. 
## Test
To test the pretrained model, run following instruction: 
```shell
python predict.py
```
## reconstructed samples
one multi-focal-plane image generated by STSRNet.
![sample](./assets/images/STSRNet_main.jpg)