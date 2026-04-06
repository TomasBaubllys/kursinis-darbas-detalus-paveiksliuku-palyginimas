#!/bin/bash

for m in {0..3}
do
	for t in {0..3}
	do
		FILENAME="./weights/mobilenetv3_bm${t}_bt${m}weights.pth"

		if [ -f "$FILENAME" ]; then
			echo "Running $FILENAME"
			python test.py -mbnet -b${t} -wf "$FILENAME" >> results.log
		fi
	done
done
