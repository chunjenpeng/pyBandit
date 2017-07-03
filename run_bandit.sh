#!/bin/bash

JOBNAME=bandit.py
totalNP=$(nproc)

if [ "$#" != "2" ]; then
    echo ""
    echo "  Usage: ./run.sh <CODE_PATH> <DATA_PATH> "
    echo "Example: ./run.sh /home/$USER/pyBandit/  /home/$USER/pyBandit/data/2017-01-01_original"
    echo ""
    exit 
fi

CODE_PATH=$1
if [ ! -d $CODE_PATH ]; then
    echo "ERROR: Cannot find directory $CODE_PATH"
    exit 1 
fi
if [ ! -f $CODE_PATH/$JOBNAME ]; then
    echo "ERROR: Cannot find file $CODE_PATH$JOBNAME"
    exit 1 
fi

DIR=$2
mkdir -p $DIR
if [ ! -d $DIR ]; then
    echo "ERROR: Cannot create directory $DIR"
    echo "       Disk probably full..."
    exit 1 
fi

echo "Using $totalNP processors..."



for DIM in 2 # 10 30 50
do
    for F in $(seq 1 25)
    do
        JOBNUM=$(ps aux | grep $JOBNAME | wc -l)
        while [ $JOBNUM -gt $totalNP ]
        do
            sleep 1
            JOBNUM=$(ps aux | grep $JOBNAME | wc -l)
        done

        cd $CODE_PATH
        python3 -u $JOBNAME $F $DIR &>  $DIR/F${F}_${DIM}D.log & 

        echo "Submitting F${F}_${DIM}D..."
        sleep 1
    done
done
