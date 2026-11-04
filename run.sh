#!/bin/bash

python marian.py ./en -o ./es \
            --model ./model \
            --device cpu \
            --compute-type int8
