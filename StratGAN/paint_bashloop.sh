#!/bin/bash


# declare an array called array and define 3 vales

# array=( one two three )
label=2
array=($(seq 0 1 30))
for i in "${array[@]}"
do
	python3 ./main.py --run_dir=line7 --gf_dim=32 --df_dim=16 --paint \
		--paint_label=$label --paint_groundtruth --paint_groundtruth_type=core \
		--paint_groundtruth_new=False --paint_groundtruth_load=4trial --paint_savefile_root="$i"_panel


	mv ./paint/line7/final.png ./paint/line7/manylines/"$label"_"$i".png
done