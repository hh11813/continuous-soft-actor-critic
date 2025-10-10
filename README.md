# Continuous Soft Actor-Critic: An Off-Policy Learning Method Robust to Time Discretization


## Environment Configuration

Create Python 3.9/3.10 Virtual Environments

```setup
conda create -p /home/myenv python=3.9/3.10
```

### Install TorchRL

You can install TorchRL from PyPi.
```setup
pip install torchrl
```
For more details, or for installing nightly versions, see the TorchRL installation guide.

### Install BenchMARL

You can clone it locally to access the configs and scripts.
```setup
git clone https://github.com/facebookresearch/BenchMARL.git
pip install -e BenchMARL
```
### Install task environments

All enviornment dependencies are optional in BenchMARL and can be installed separately.

### Install VMAS

You can clone it locally to access the scripts.
```setup
git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git
pip install -e VectorizedMultiAgentSimulator
```

### Install marl-eval
```setup
git clone https://github.com/instadeepai/marl-eval.git
pip install -e marl-eval
```
or 
```setup
pip install id-marl-eval
```


