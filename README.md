<p align="center">
  <a href="#">
    <img src="https://img.shields.io/github/license/itaigat/pytorch-template" alt="PyPI - License" />
  </a>
    <a href="https://lgtm.com/projects/g/itaigat/pytorch-template/context:python">
        <img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/itaigat/pytorch-template.svg?logo=lgtm&logoWidth=18"/>
    </a>
    <a href="https://lgtm.com/projects/g/itaigat/pytorch-template/alerts/">
        <img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/itaigat/pytorch-template.svg?logo=lgtm&logoWidth=18"/>
    </a>
</p>

----------------------

# PyTorch template
In this project, we provide a strong template for a PyTorch project.
The purpose of this repository is to provide an example (and strong utils on the way) for a deep learning project using
PyTorch.
## TensorBoard
To run tensorboard simply run in the project directory `tensorboard --logdir logs/tensorboard`
## Structure
### Configuration
We use the [Hydra](https://github.com/facebookresearch/hydra) framework for configuration. It allows us to easily read
configuration files and to do hyper-parameter tuning.

The configuration file is stored under `config/config.yaml` and has many parameters. At the beginning of each experiment,
the configuration file is validated with a schema. The schema is stored at `utils/config_schema.py`.

In case we want to add/remove a parameter from the configuration file we need to:
* Change the YAML file
* Change the schema at `utils/config_schema.py`
* Verify that we don't break anything in `TrainParams`

To run hyperparameter tuning just follow the instructions
[here](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).
### Models
The model package should store all the models we use. You can take a look at `models/base_model.py` in `MyModel`
for example.
### Networks
The networks package should hold layers such as modified sequential. For example, in the `nets` package, you can see
`FCNet` which is an easy implementation for a fully connected network with weight normalization, dropout and easy way to
add hidden layers with different values of hidden neurons.
### Train
The train function gathers all the training logic. Its structure is built for an easy change.
For example, you can change the optimizer and more.

Moreover, during the training and evaluation stage, the logger reports relevant metrics to stdout, file, and TensorBoard.
### Dataset
New dataset creation built from three stages:
1. Set variables - set the relevant inputs to self.
2. Load features - pick the best way to load features (memory/disk) and implement the load stage under `self._get_features()`
3. Create a list of entries - implement `self._get_entries()`

Then, in `__getitem__` you only need to retrieve samples from the list you created in stage #3.
### Logger
The logger class is logging messages to:
1. Tensorboard
2. Stdout
3. Files

Each experiment has a directory under `logs/` (configurable).
By default, the best epoch and all the output will be saved there.
### Types
In case you add/change a type you can add it to `utils/types.py`
## Checklists
### Checklist for a new dataset
To add a new dataset:
- [ ] Add relevant variables to the constructor
- [ ] Implement `self._get_features`
- [ ] Implement `self._get_entries`
- [ ] Implement `self.__getitem__`
### Checklist for a change in configuration
To change a variable in the configuration file:
- [ ] Change it in the `config.yaml` file
- [ ] Update the schema under `utils/config_schema.py`
- [ ] Update `TrainParams` in `utils/train_utils.py`
## Credits
* [Hydra](https://github.com/facebookresearch/hydra) for a great configuration framework.
* [bottom-up-attention-vqa](https://github.com/hengyuan-hu/bottom-up-attention-vqa) for great modified layers.