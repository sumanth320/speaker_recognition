# Speaker Recognition
To identify the Speaker by extracting 10 features to identify an Audio Clip by it's features. These features are clubbed in an appropriate
format and fed to the ML model to train to recognize the Speaker.

## System Requirements
Make sure you have the latest Python, CUDA and NVIDIA Drivers for proper installation and smooth execution
IDE : Anaconda

## Installation
	- Create a virtual enviornment: This is a good practice but not necessary
	- Goto Anaconda Prompt and type in the below command. This command will create a virtual enviornment.
	  conda create -y --name tensorflow python=3.7
	- Now activate the virtual enviornment, by executing the below command. This command will change the prompt from "base" to 
	  "tensorflow"
	- Activate tensorflow

- In the Terminal type the following commands to install the libraries,this will check the exsisiting version and will install the latest
  one.
	- conda install -y scipy
	- pip install --exists-action i --upgrade tensorflow==2.0.0
	- pip install --exists-action i --upgrade keras
	- pip install --exists-action i --upgrade matplotlib
	- pip install --exists-action i --upgrade sklearn
	- pip install --exists-action i --upgrade pandas
	- pip install cv2
	- pip install librosa
	- conda install -c conda-forge python-sounddevice

- Check the version installed using below commands:
	- tf.__version__  	(where tf is the alias of tensorflow)
	- k.__version__		(where k is the alias of keras)
	- sys.version		
	- pd.__version__	(where pd is the alias of pandas)
	
- Select the virtual Environment:
	- In Anaconda Navigator --> Home, Click on the "Applications on" dropdown and Select "tensorflow".
	- This will take a few minutes, after that install the prefered IDE (Jupyter, Spyder),installation is required for first 
time only
	- Launch the IDE and start coding.
	
## Possible Errors

```
"tensorboard 2.0.0 has requirement setuptools>=41.0.0, but you'll have setuptools 40.8.0 which is incompatible."

This is a possible error which can be solved using the command 

pip install --upgrade --user tensorflow

```
"PackagesNotFoundError: The following packages are not available from current channels:"

While installing sounddevice it is advisable to get the commands directly from anaconda.org by creating an account in Anaconda Cloud.
These commands would avoid that error.

## License
Open Source Software

##General Instructions

- Make sure proper libraries are imported while executing and please follow best coding practices.

- The modules in project are independent of each other.(IMPORTANT)

Happy Coding!

