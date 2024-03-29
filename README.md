## Deep Generation of UI Inputs (Deep-GUI)

Deep-GUI is a tool for generating intelligent inputs to test UI-based applications, such as Android or web applications.
This document provides instruction on how to train a Deep-GUI model, how to use it in order to  automatically 
interact with an application, and how to quantify the quality of the generated inputs in terms of line coverage.

## Prerequisites

System requirements:
* Ubuntu 18.04
* Python 3.6.8
* Java 8

Before getting started:
* Install Android SDK
	* Download and extract [Android SDK commandline tool](https://developer.android.com/studio?gclid=CjwKCAjwsNiIBhBdEiwAJK4khrwyna4oZ6jQkkxhGoo-uwrnpyDt3BJkkd0Bb2Hy01QzyOps16QByBoCKFUQAvD_BwE&gclsrc=aw.ds#command-tools) in `~/android-sdk`. Locate `sdkmanager` in `~/android-sdk/cmdline-tools/bin`    
	* Install build-tools 30.0.3 using `sdkmanager`
	* Install system image `system-images;android-10;google_apis;x86` using `sdkmanager`
* Install Mozilla Firefox for testing web applications
	* install [geckodriver v0.27.0](https://github.com/mozilla/geckodriver/releases/tag/v0.27.0) and add it to `PATH`
* Clone this repository in `/home/$USER`
	* Install Packages in requirements.txt

**Note: it is important to install the requirements in the exact paths mentioned above.**
## Repository Structure
```
deep-gui        
│
└───apks: a sample set of apk files with their respective emma file.
│
└───configs: configuration templates for different tasks
│   
└───models: pre-trained deep-gui models
│   
└───scripts: bash scripts used by different parts of the tool
│   
└───src: python source files 
```

## Usage
You can use this software to perform these tasks:

### Data Collection
In order to train deep-gui, you need to first collect data.

1. Create emulator template
   ```
   cd scripts
   ./create_tester_ref
   ./clone_avd.sh tester_ref collector_ref
   ```
2. Set the configurations:
	
   In `configs/collect-configs.yaml`, set these mandatory values:
   ```
   collectors: The first number is the number of parallel data collection agents
   data_file_dir: The directory where the data is stored in (change this dircetory to collect both training and validation data)
   logs_dir: The directory where the tensorboard logs are written to
   collectors_apks_path: The directory containing training/validation apks
   collector_configs.version_start: If you need to append to existing data, set this number accordnigly
   ```
3. Copy `configs/collect-configs.yaml` to `src/configs.yaml`
4. Run the code: `cd src; python main.py`
5. You can also monitor the progress:
	```
	cd <logs_dir> # same as logs_dir used in configs.yaml file
	tensorboard --logdir=. --reload_interval 1 --samples_per_plugin "images=0"
	```
    Connect to `localhost:6006` to see tensorboard logs.


### Training
To train using the collected data:
1. Set the configurations:
	
   In `configs/train-configs.yaml`, set these mandatory values:
   ```
   data_file_dir: The directory containing the training data
   learner_configs.save_dir: The directory where the trained models are stored. You need to create this directory manually.
   learner_configs.validation_dir: The directory containing the validation data
   collector_configs.version_start: Set this to a large number
   ```
2. Copy `configs/train-configs.yaml` to `src/configs.yaml`
3. Run the code: `cd src; python main.py`

<b>Note:</b> A pre-trained model is availabe in `models`

### Android Experiments (Monkey++)
To run the experiments:
1. Create emulator template
   ```
   cd scripts
   ./create_tester_ref
   ./clone_avd.sh tester_ref collector_ref
   ```
2. Set the configurations:
	
   In `configs/monkey-test-configs.yaml`, set these mandatory values:
   ```
   testers: The first number is the number of parallel agents.
   weights_file.e10: The path to the model that is to be used (.hdf5)
   ```
3. Copy `configs/monkey-test-configs.yaml` to `src/configs.yaml`

4. Run the code:
	* If you want to run monkey without deep-gui: 
		```
    	cd scripts
		./run_all_monkies.sh <experiment-name> monkey 0 <num-agents> <num-rounds> <apk-dir> <experiment-dir>
		```
    * If you want to run deep-gui: 
		```
    	cd scripts
		./run_all_monkies.sh <experiment-name> deep 1 <num-agents> 1 <apk-dir> <experiment-dir>
		```
 	where:
    ```
   <experiment-name>: An arbitrary name for the experiment
   <num-agents>: Number of parallel agents (must match the configs.yaml file)
   <apk-dir>: The directory containing test apks. Each apk named app.apk needs to have a emma file in the same directory named app.apk.em
   <experiment-dir>: The directory containing the experiment files
   ```
5. After the experiment is completed, run this:
	```
	cd ../src
	python update_tb.py <experiment-dir> <experiment-name> <experiment-dir>/tb_otest_logs <apk-dir>
	```

6. Look at the logs:
	```
	cd <experiment-dir>/tb_otest_logs
	tensorboard --logdir=. --reload_interval 1 --samples_per_plugin "images=0"
	```
    Connect to `localhost:6006` to see tensorboard logs.


### Web Experiments
To run the experiments:
1. Set the configurations:
	
    In `configs/web-configs.yaml`, set these mandatory values:
    * If you want to run a random agent:
    	```
        testers: Set to [<num-parallel-agents>, [0, 0, 0, 0, 0, 1], monkey]
        reward_predictor: [RandomRewardPredictor, random]
        ```
    * If you want to use deep-gui:
    	```
        testers: Set to [<num-parallel-agents>, [.7, .3], deep, c99s, e10]
        reward_predictor: [UNetRewardPredictor, unet]
        ```
    ```
    logs_dir: The directory where the tensorboard logs are written to
    weights_file.e10: The path to the model that is to be used (.hdf5)
    browser_configs.apps: The list of websites to be explored
    ```
2. Copy `configs/web-configs.yaml` to `src/configs.yaml`
3. Run the code: `cd src; python main.py`
4. You can also monitor the progress:
	```
	cd <logs_dir> # same as logs_dir used in configs.yaml file
	tensorboard --logdir=. --reload_interval 1 --samples_per_plugin "images=0"
	```
    Connect to `localhost:6006` to see tensorboard logs.

### Analysis
To analyze the results:
1. Uncomment the appropriate section in `scripts/analysis.sh`
2. Run the analysis:
	```
	cd scripts
	./run_all_analyses.sh <experiment-dir> <apk-dir> <num-agents>
	```
3. Check the results:
	```
	cd <experiment-dir>/tb_otest_logs/analysis
	tensorboard --logdir=. --reload_interval 1 --samples_per_plugin "images=0"
   	```
    
	Connect to `localhost:6006` to see tensorboard logs.


## References
If you use the code or data, please cite the following paper: 
```
@INPROCEEDINGS{9678778, 
author={YazdaniBanafsheDaragh, Faraz and Malek, Sam},
title={Deep GUI: Black-box GUI Input Generation with Deep Learning},
booktitle={2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
year={2021},  volume={},  number={},  pages={905-916},  doi={10.1109/ASE51524.2021.9678778}}
```
