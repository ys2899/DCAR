#!/bin/sh


if [ -d "./training_history" ]
then
    echo "Directory training_history exists."
else
    mkdir ./training_history
fi


now=$(date +%Y%m%d%H%M%S)


logfile=../training_history/mylogfile_$now


mkdir $logfile


find -iname '*.py' -exec cp {} $logfile \;


nohup python dcARrnn_train.py --config_filename=data/model/dcrnn_la.yaml > $logfile/log.out 2>&1 &