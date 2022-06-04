#!/bin/bash

jupyter nbconvert --to notebook --inplace --execute $1

jupyter nbconvert --to html $1
