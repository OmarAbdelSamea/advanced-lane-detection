#!/bin/bash

DEBUG_VAR=0
OUTPUT_VID=0
INPUT_VID=0

if [[ $# -eq 0 || ($# -eq 1 && ${1::1} == "-") ]] ; then
        echo "[ERROR] No input file specfied"
        echo "Usage: [-d] /path/to/input-video.mp4 /path/to/output-video.mp4"
        exit 1
fi 

while getopts "hd" OPTION; do
        case $OPTION in
                d)
                        DEBUG_VAR="1"
                        ;;
                h)
                        echo "Usage:"
                        echo "advanced-lane.sh /path/to/input-video.mp4 /path/to/output-video.mp4 "
                        echo ""
                        echo "   -d     to enable debugging mode"
                        echo "   -h     help (this output)"
                        echo ""
                        exit 0
                        ;;
                        
                [?])	
                        echo "Usage: [-d] /path/to/input-video.mp4 /path/to/output-video.mp4"
                        exit 1
                        ;;
        esac
done
if [[ $1 == "-d" ]] ; then
        INPUT_VID=$2
        if [[ -f $2 &&  -n $3 ]] ; then
                OUTPUT_VID=$3
        elif [[ -f $2 ]] ; then
                OUTPUT_VID="lane_detection.mp4"
        fi  
else
        INPUT_VID=$1
        if [[ -f $1 &&  -n $2 ]] ; then
                OUTPUT_VID=$2
        elif [[ -f $1 ]] ; then
                OUTPUT_VID="lane_detection.mp4"
        fi  
fi  

python3 lane.py $INPUT_VID $OUTPUT_VID $DEBUG_VAR 