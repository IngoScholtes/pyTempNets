#!/bin/bash

# NOTE doing every test on its own is much easier as finding out what python 
# NOTE interally does when running multiple test

for my_test in $(ls *.py); do
	python -m unittest discover -p "$my_test"
done

