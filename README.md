# Simple implementation of Echo State Network with NARMA benchmark task
basic echo state network implementation

## problem setting
reimplementation of simple echo state network on NARMA-10 task.
problem setting is from [this paper](http://www.deepscn.com/pdfs/Jaeger_2002.pdf)

## hyper parameters
All the hyper parameters can be controlled by ```config/baseline.yaml```. If you want to try with other sets of parameters, all you need to do is modifying that yaml file.

As `hydra`, python library for hyper parameter control, is used, all the output of the program is going to `baseline/` directory, specified with `hydra:run:dir` of yaml file. 

## code explanation
- esn_forward.py: calculate the states of echo state network
- narma_data_gen.py: code to generate narma dataset
- esn_offline.py: training and inference code for offline dataset 
- main_offline.py: main code integrating everything above

## how to run
### necessary libraries
see `requirements.txt`
### environment variables
add `/path/to/esn_basic` to `$PYTHONPATH`