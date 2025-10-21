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

## Reproduce Experiments

To Reproduce Experiments in the Paper:​

1. Replace the `__init__.py` File​​:

Replace the existing `/BenchMARL/benchmarl/algorithms/__init__.py` file with the one provided by the authors.

​​Key modification​​: This file adds the ​​CMASAC algorithm class (Test)​​ and its configuration (​​TestConfig).

2. Add the author-provided `test.py` and `test.yaml` files to the following directories, respectively:

`​​test.py`​​: `/BenchMARL/benchmarl/algorithms/`

​​`test.yaml`: `/BenchMARL/benchmarl/conf/algorithm/`

`test.py` implements the CMASAC algorithm, and `test.yaml` is the configuration file for CMASAC.​

3. Replace the existing `/BenchMARL/benchmarl/conf/experiment/base_experiment.yaml` file with the provided `base_experiment.yaml​`​ or adjust the hyperparameters according to the configurations.

4. Replace the existing `/VectorizedMultiAgentSimulator/vmas/simulator/core.py` file with the ​​author-provided `core.py​​`, or directly modify the ​​time discretization parameter $\delta t$​​ to:
```setup
dt: float = 0.01,  # 0.1,   
```
5. Add the provided `plot.py`, files to the directories `/BenchMARL/examples/plotting/​`

## Training and Evaluation
To train and evaluate the model(s) in the paper, run this command:

```train
python BenchMARL/examples/plotting/plot_navigation.py
```
The script is configured to run on Linux systems by default. If you are executing it on a Windows system, please replace the corresponding paths.
## Technical Guidelines

### ​​Adjust Time-Step for Simulations​​
If comparing performance across different ​​discrete time-steps​, ensure to modify the time-step parameter in `core.py` (simulation environment) :
```setup
dt: float = 0.01,  # 0.1.  
```
and "dt" in "test.yaml", which refers to the variation.

### ​​Validate CMASAC/TEST Performance​​

To validate the performance of ​​CMASAC​​ or ​​TEST​​ (with non-scaled parameters as described in the paper), adjust the following terms in `test.py`:

​​Line 916​​: `loss_actor`;

​​Line 957​​: `pred_val`, `target_value`.

Adjust the ​​learning rates​​ in `base_experiment.yaml` to match the experimental setup.

### ​​Adjustment of reward function

We can modify the reward function in the corresponding simulation environment in `VectorizedMultiAgentSimulator/vmas/scenarios` by scaling it with `1/dt` (where `dt` is defined in `test.yaml`), while simultaneously scaling the hyperparameters in `base_experiment.yaml​`.

