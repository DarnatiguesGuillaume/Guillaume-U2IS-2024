#!/bin/bash
#./process.sh 50 2 8 36 1 ror


./process.sh 50 1 8 10 1 uniform





: << 'COMMENT_BLOCK'
This comment is just for me to remember how to do a loop in bash...

values=(45 55)

for value in "${values[@]}"; do
    ./process.sh "$value" 2 8 36 1 ror
done
COMMENT_BLOCK