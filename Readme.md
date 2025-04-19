# DA6401-Assign-2
## Nikshay Jain | MM21B044

This project focuses on training & evaluating CNN models for the CIFAR-10 dataset. The project is divided into 2 parts - for a custom model training and fine-tuning.

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [Setup Instructions](#setup-instructions)
3. [Usage](#usage)
4. [Closing Note](#closing-note)

## Directory Structure
```bash
DA6401-Assign-2
├── inaturalist/                            # Original raw dataset directory
│   ├── train/                              # train dir -> splitted into train and val in code
│   └── val/                                # val dir -> used for test
├── .venv                                   # Virtual env
├── Part-A/                                 # Python scripts for part A
│   ├── A_classes.py                        # Contains code for all the classes necessary
│   ├── A_main.py                           # main function for final execution
│   ├── A_sweep.py                          # conatins all the functions needed for sweeping across configs
│   ├── A_train_test.py                     # for training and testing of the final model after sweeps.
│   └── da6401-assign-2-Part-A.ipynb        # Kaggle notebook used to run the code for Part A
├── Part-B/                                 # Python scripts for part B
│   ├── B_classes.py                        # Contains code for all the classes necessary
│   ├── B_main.py                           # main function for final execution
│   ├── B_main_models.py                    # function to compare various models keeping startegies fixed.
│   ├── B_funcs.py                          # this has all the functions necessary to run the programs
│   └── da6401-assign-2-Part-B.ipynb        # Brownie task to get the class distributions
├── .gitignore                              # gitignore file
├── da6401-assign-2.ipynb                   # Full Kaggle notebook
├── Assignment sheet.pdf                    # Assignment pdf
├── requirements.txt                        # Libraries list
└── Readme.md                               # Project documentation
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nikshay-Jain/DA5402-Assign-5.git
   cd DA6401-Assign-2
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.11+ installed to the required libraries after creating and activating a virtual env in python:

   ```bash
   pip install -r requirements.txt
   ```
   This file installs the libraries used by the code:

## Usage
Before any usage, just go to the **respective main function file** and fill in your **API key** in the variable named `key`. This would help in logging into wandb servers. Please note, it is to be done for all the main files.

### Part A:
Run the script by executing this command in the terminal
```bash
python Part-A\A_main.py
```

Next, it asks you if you want to run the sweeps (y/n):
- for testing the sweeps part, type 'y' and the run_sweeps function is triggered for the same.
- if you want to go for analysing the runs already made, just go for 'n' and the `train_best_model` is triggered to run to get the best possible configuration from wandb and generate the needed performance metrics.

### Part B:
Run the script by executing this command in the terminal
```bash
python Part-B\B_main.py
```
This compares various finetuning strategies for resnet50 and evaluates them over test data.

However, if you want to compare models like efficentnet, vgg16, vit_b_16, you can go for running
```bash
python Part-B\B_main_models.py
```

Both of these would ask you for the wandb.ai API key which you have to feed in for getting connected to the servers which can be entered as mentioned above.

## Closing Note:
The link to the GitHub repo is attached here: https://github.com/Nikshay-Jain/DA6401-Assign-2

The link to the wandb.ai Report is attached here: https://api.wandb.ai/links/mm21b044-indian-institute-of-technology-madras/ekxt7nnq