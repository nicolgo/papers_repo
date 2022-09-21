# Reproduce steps
## 1 prepare dataset
### 1.1 image dataset
on Windows, download dataset from this [URL](https://drive.google.com/file/d/17CwkBPtSX92PYz_XMLGxxjjavw_DeeuL/view?usp=sharing). the origianl url of this dataset is from this [repos](https://github.com/fastai/imagenette). As for the file structure, which is same as the previoud work [latent-diffusion](https://github.com/CompVis/latent-diffusion)

## 2 set configuration
### 2.1 first stage: training autoencoder
1. run the `main.py` with following parameters
```
--base configs/autoencoder/autoencoder_kl_64x64x3.yaml -t --gpus 0,
```
the parameter of autoencoder is saved in the `yaml` config file:
```
 # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
```
