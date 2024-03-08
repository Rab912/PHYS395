#!/bin/bash

echo "Running \"problem_1.py\"..."
python3 problem_1.py

echo "Running \"problem_3.py\"..."
python3 problem_3.py

if [ $# -eq 0 ]
    then
        echo "No file was supplied for \"problem_5.py\". Terminating..."
        exit
fi

echo "Running \"problem_5.py\"..."
python3 problem_5.py $1

echo "Done"