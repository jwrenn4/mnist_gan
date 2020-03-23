# MNIST GAN

This repository contains some small utilities to get started creating a generative adversarial network (GAN) to create synthetic handwritten digits.

## Setup

This repo has a few scripts in it, as well as some already-built models.  The most important script is the gan.py file, which is a Python 3 file (has successfully been run in python 3.7.6) with a command line interface to create GAN models.

Assumptions in setup:

1. You have a functional python environment (preferrably Anaconda) with pip installed
2. You are working in a linux (or linux-like - I am working on a Windows machine inside the Windows Subsystem for Linux for my command line operations) operating environment with git installed

Running the following commands should get everything set up for you:

```bash
# Clone the repository

cd desired-location-to-place-repository-into
git clone https://github.com/jwrenn4/mnist_gan.git
cd mnist_gan

# Install the requirements from the requirements.txt file
pip install -r requirements.txt
```

## Running

There are two ways to run the system to create models and generate fake images.  The first one, which is completely automated for all digits, is to run the Run.sh script.  This script is written in bash, and it creates models and images for each digit.  Simply run the following and sit back:

```bash
./Run.sh ./gan.py ./images ./models
```

The second way is to call the gan.py script individually for each number.  This method allows the user to change hyperparameters which are not available when calling the full script.  To pull up the help documentation for the script, run:

```bash
python gan.py --help
```

This should be enough to get you on your way.