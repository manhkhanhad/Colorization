<h1 align="center">Welcome to Deoldify Colorization 👋</h1>
<p>
  <a href="docs" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.5.0-blue.svg?cacheSeconds=2592000" />
  <a href="abc" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://twitter.com/manhkhanhad" target="_blank">
    <img alt="Twitter: manhkhanhad" src="https://img.shields.io/twitter/follow/manhkhanhad.svg?style=social" />
  </a>
</p>

> Reimplementaion of Deoldify base on pixpix colorization
### 🏠 [Homepage](https://github.com/manhkhanhad/Colorization)
### ✨ Demo: Comming soon

## Install

```sh
pip install fastai==2.4.0
pip install -r requirement.txt
```

## Train
Pretrain Generator
```sh
python pretrain.py --dataroot path/to/dataset --checkpoints_dir path/to/checkpoint --model deoldify --name name_model --n_epochs num_epoch --n_epochs_decay num_epoch_decay_lr --batch_size batchsize 
```
Train Generator and Discriminator
```sh
python train.py --dataroot path/to/dataset --checkpoints_dir path/to/checkpoint --model deoldify --name name_model --n_epochs num_epoch --n_epochs_decay num_epoch_decay_lr --batch_size batchsize 
```
## Run tests
Download the checkpoint from [checkpoints](https://drive.google.com/file/d/1OiXeWpDszWtBM_1dfZJdwUziLLU_Co-l/view?usp=sharing), unzip and 
put the folder under ./checkpoints
```sh
python test.py --dataroot path/to/dataset --checkpoints_dir checkpoints --model deoldify --name name_model --batch_size batchsize --num_test number_of_test_image --results_dir path/to/results
```

## Author

👤 **manhkhanh**

* Website: manhkhanhad.github.io
* Twitter: [@manhkhanhad](https://twitter.com/manhkhanhad)
* Github: [manhkhanhad](https://github.com/manhkhanhad)
* LinkedIn: [@Khanh Ngo Huu Manh](https://www.linkedin.com/in/khanh-ngo-huu-manh-29b73519a/)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/manhkhanhad/Colorization/issues). 

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
