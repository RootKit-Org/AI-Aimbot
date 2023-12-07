```markdown
# Installing Requirements

Follow these steps to install all the requirements to your system:

## Step 1: Activate your environment:

## Step 2: Only if you have an NVIDIA graphics card - Download and Install CUDA:

Nvidia CUDA Toolkit 11.8 [DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

## Step 3: Install PYTORCH:

- For NVIDIA GPU:

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

- For AMD or CPU only:

pip install torch torchvision torchaudio

## Step 4: Install requirements.txt:

pip install -r requirements.txt

## Step 5: Install additional modules:

Because you are using Conda, you need to install additional requirements in your environment.

pip install -r Conda/additionalRequirements.txt

## Step 6: Test your installation:

To test your installation, run the following command:

python main.py

You should now have a working AI AIMBOT. If you want to use the fastest version continue the installation steps on the RootKit AI Aimbot README.md
```
