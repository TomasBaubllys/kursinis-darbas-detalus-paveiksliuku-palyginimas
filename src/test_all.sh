#!/bin/bash

for m in {0..2}
do
	for t in {0..3}
	do
		FILENAME="mobilenetv3_botm${t}_bott${m}.pth"

		if [ -f "$FILENAME" ]; then
			echo "Running $FILENAME"
			python test.py -mbnet -b${t} -wf "$FILENAME" >> results.log
		fi
	done
done
