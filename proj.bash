#!/bin/bash

DEBUG_VAR=0
OUTPUT_VID=0
INPUT_VID=0
MODE="--both"
if [[ $# -eq 0 || ($# -eq 1 && ${1::1} == "-") ]] ; then
        echo "[ERROR] No input file specfied"
        echo "Usage: [-y/-n/-d/-t] /path/to/input-video.mp4 /path/to/output-video.mp4"
        exit 1
fi 

while getopts "hdnyt" OPTION; do
        case $OPTION in
                t)
                        MODE="--yolo-tiny"
                        ;;
                y)
                        MODE="--yolo"
                        ;;
                n)
                        MODE="--lane-no-debug"
                        ;;
                d)
                        MODE="--lane-debug"
                        ;;
                h)
                        echo "Usage:"
                        echo ".sh /path/to/input-video.mp4 /path/to/output-video.mp4 "
                        echo ""
                        echo "   -t     car detection only with tiny yolo"
                        echo "   -y     car detection only with yolo"
                        echo "   -n     lane detection with no debugging"
                        echo "   -d     lane detection with debugging"
                        echo "   -h     help (this output)"
                        echo ""
                        exit 0
                        ;;
                        
                [?])	
                        echo "Usage: [-y/-n/-d/-t] /path/to/input-video.mp4 /path/to/output-video.mp4"
                        exit 1
                        ;;
        esac
done
if [[ $1 == "-d" || $1 == "-y" || $1 == "-n" || $1 == "-t" ]] ; then
        INPUT_VID=$2
        if [[ -f $2 &&  -n $3 ]] ; then
                OUTPUT_VID=$3
        elif [[ -f $2 ]] ; then
                OUTPUT_VID="video_output.mp4"
        fi  
else
        INPUT_VID=$1
        if [[ -f $1 &&  -n $2 ]] ; then
                OUTPUT_VID=$2
        elif [[ -f $1 ]] ; then
                OUTPUT_VID="video_output.mp4"
        fi  
fi  

python main.py $MODE $INPUT_VID $OUTPUT_VID 
