#!/bin/bash

if [ $# -eq 0 ]
    then
        echo "No file was supplied. Terminating..."
        exit
fi

echo "Running \"problem_1.py\"..."
python3 problem_1.py $1

echo "Running \"problem_2.py\"..."
python3 problem_2.py $1

echo "Running \"problem_3.py\"..."
python3 problem_3.py $1

echo "Running \"problem_4_1.py\"..."
python3 problem_4_1.py $1

echo "Running \"problem_4_2.py\"..."
python3 problem_4_2.py $1

echo "Done"