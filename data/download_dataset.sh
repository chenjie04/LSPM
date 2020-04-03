#!/bin/bash

printf "Downloading the ml-1m dataset.....\n\n"


wget -c http://files.grouplens.org/datasets/movielens/ml-1m.zip

unzip -q ml-1m.zip

printf "Done!\n"

