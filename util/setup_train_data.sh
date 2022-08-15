#!/bin/bash


declare -a pairs=("ar_ke" "ke_ke")

# sim < 0.65

# ar_ke

PRE=''

cp ${PRE}test.ke test.ar_ke.ke
cp ${PRE}test.ar test.ar_ke.ar

cp ${PRE}valid.ke valid.ar_ke.ke
cp ${PRE}valid.ar valid.ar_ke.ar

cp ${PRE}train.ke train.ar_ke.ke
cp ${PRE}train.ar train.ar_ke.ar

# ke_ke

cp ${PRE}test.ke test.ke_ke.ke
cp ${PRE}test.ke test.ke_ke.ar

cp ${PRE}valid.ke valid.ke_ke.ke
cp ${PRE}valid.ke valid.ke_ke.ar

cp ${PRE}train.ke train.ke_ke.ke
cp ${PRE}train.ke train.ke_ke.ar







