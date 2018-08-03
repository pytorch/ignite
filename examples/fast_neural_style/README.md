# fast-neural-style

### Introduction
This example is ported over from [pytorch-examples](https://github.com/pytorch/examples/tree/master/fast_neural_style).

It uses pytorch-ignite to implement an algorithm for artistic style transfer as described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

This repo gives examples on how to create a Progress Bar with `pytorch-ignite`. 

### Requirements

The code can be executed using both `Python 2.7` and `Python 3.5`.

* `torch`
* `torchvision`
* `pytorch-ignite`

Example for `virtualenv` setup:

`virtualenv --python=/usr/bin/python3.5 env`

`source env/bin/activate`

`pip install torch torchvision pytorch-ignite`

The code runs on CPU, but GPU allows it to run much faster. If using GPU, please ensure proper libraries are installed. 

### Documentation

#### Training
Code can be used to train a style transfer model for any image. To run code correctly, ensure that [MSCOCO dataset](http://images.cocodataset.org/zips/train2014.zip) and a style image are downloaded.

Since the code using Pytorch's Dataset functions, ensure that directory with MSCOCO dataset is formatted as shown below. The directory should be setup such that the location of the dataset is MSCOCO, which contains a single folder 0, containing all the images. 

MSCOCO

├── 0                    
│   ├── RY48TY43YT.jpg          
│   ├── 4324J0FNFL.jpg      
│   └── Y9REWJKNFE.jpg                
└

##### Example
`python examples/fast_neural_style/neural_style.py train --epochs 2 --cuda 1 --style_image ./images/style_images/mosaic.jpg --test_image ./images/content_images/amber.jpg`

##### Flags
* `--epochs`: number of training epochs, default is 2. 
* `--batch_size`: batch size for training, default is 32".
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. 
* `--style_image`: path to style-image. 
* `--test_image`: path to test-image, stylized at the end of each epoch. 
* `--checkpoint_model_dir`: path to folder where checkpoints of trained models will be saved. 
* `--checkpoint_interval`: number of batches after which a checkpoint of trained model will be created. 
* `--stylized_test_dir`: path to folder where stylized test image will be saved at end of each epoch. 
* `--image_size`: size of training images, default is 256 X 256.
* `--style_size`: size of style-image, default is the original size of style image. 
* `--cuda`: set it to 1 for running on GPU, 0 for CPU. 
* `--seed`: random seed for training. 
* `--content_weight`: weight for content-loss, default is 1e5. 
* `--style_weight`: weight for style-loss, default is 1e10. 
* `--lr`: learning rate, default is 1e-3. 


#### Evaluation

Code can be used to stylize an image using a trained style transfer model. 

##### Example
`python examples/fast_neural_style/neural_style.py eval --content_image ./images/content_images/amber.jpg --output_image test.png --cuda 1 --model ./checkpoints/checkpoint_net_1.pth`

#### Flags
* `--content_image`: path to content image you want to stylize.
* `--content_scale`: factor for scaling down the content image.  
* `--output_image`: path for saving the output image.  
* `--model`: saved model to be used for stylizing the image.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU. 
