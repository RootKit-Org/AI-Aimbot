```markdown
# Installing Miniconda

Follow these steps to install Miniconda on your system:

## Step 1: Download Miniconda

First, download the appropriate Miniconda installer for your system from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

## Step 2: Run the Installer

- **Windows**: Double click the `.exe` file and follow the instructions.
- **macOS and Linux**: Open a terminal, navigate to the directory where you downloaded the installer, and run the following command:
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Replace `Miniconda3-latest-Linux-x86_64.sh` with the name of the file you downloaded.

## Step 3: Follow the Prompts

The installer will prompt you to review the license agreement, choose the install location, and optionally allow the installer to initialize Miniconda3 by appending it to your `PATH`.

## Step 4: Verify the Installation

To verify that the installation was successful, open a new terminal window and type:

```bash
conda list
```

If Miniconda has been installed and added to your `PATH`, this should display a list of installed packages.

## Step 5: Update Conda to the Latest Version

It's a good practice to make sure you're running the latest version of Conda. You can update it by running:

```bash
conda update conda
```

That's it! You have successfully installed Miniconda on your system.

Now when you open up a terminal you should see a prompt and (base) to indicate no conda environment is active.

![Your console](imgs/console.jpg)
