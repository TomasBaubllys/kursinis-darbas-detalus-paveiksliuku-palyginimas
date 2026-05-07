# Optimizing Lightweight Neural Networks for Person Reidentification
This repository consists of all the code used to run experiments for my course work for Vilnius University Computer Science program.

## Creating the enviroment
To create the enviroment to run the programs, you must first create a Docker image.
```
bash create_d_img.sh
```
Altearnatively you can install the python packages in src/requirements.txt and run without docker

## Entering the enviroments
To enter the docker container you can run the script
```
bash start_d_cont.sh
```

## Running the tests
```
python test.py [-mbnet] -wf [WEIGHTS FILE] -b [OPTIMIZATION] 
```
Use the flag -mbnet with mobilenetv3 weight files. For the -b flag always provide the number after "bm" in the weights file.

## Training the models yourself
If you wish to train the models yourself, you can run
```
python train.py -h
```
to see all the available training options.

