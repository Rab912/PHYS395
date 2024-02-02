#!/bin/bash

echo "Enter a data file name:"

read FILE <&0

echo "Running \"problem_1.py\"..."
./problem_1.py < $FILE

echo "Running \"problem_2.py\"..."
./problem_2.py < $FILE

echo "Running \"problem_3.py\"..."
./problem_3.py < $FILE

echo "Running \"problem_4_2.py\"..."
./problem_4_2.py < $FILE

echo "Running \"problem_4_3.py\"..."
./problem_4_3.py < $FILE

echo "Done"