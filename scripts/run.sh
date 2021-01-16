#!/bin/bash

JOBNAME=run.py
# nproc = number of processors on your machine
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

DATA_PATH=$2
mkdir -p $DATA_PATH
if [ ! -d $DATA_PATH ]; then
    echo "ERROR: Cannot create directory $DATA_PATH"
    echo "       Disk probably full..."
    exit 1 
fi

echo "Using $totalNP processors..."

for DIM in 2 #10 30 50
do
    for ALGO in "ACOR" #"PSO" "CMA"
    do
        DIR=$DATA_PATH/$ALGO
        mkdir -p $DIR
        for F in $(seq 1 25)
        do
            for repeat in $(seq 1 25)
            do
                JOBNUM=$(ps aux | grep $JOBNAME | wc -l)
                while [ $JOBNUM -ge $totalNP ]
                do
                    sleep 1
                    JOBNUM=$(ps aux | grep $JOBNAME | wc -l)
                done
    
                cd $CODE_PATH
                # Run bandit
                python -u $JOBNAME -b True -a $ALGO -i $F -d $DIM -v True -csv $DIR/F${F}_${DIM}D_${repeat}.csv &>  $DIR/F${F}_${DIM}D_${repeat}.log & 


                # Run original Algo
                #python -u $JOBNAME -a $ALGO -i $F -d $DIM -v True -csv $DIR/F${F}_${DIM}D_${repeat}.csv &>  $DIR/F${F}_${DIM}D_${repeat}.log & 
    
                echo "Submitting $ALGO F${F}_${DIM}D_${repeat}..."
                sleep 1
            done
        done
    done
done
