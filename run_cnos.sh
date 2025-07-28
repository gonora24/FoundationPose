OBJECT="T"
OUTPUT_DIR="/home/pano/hiwi/nora/FoundationPose/demo_data/$OBJECT/cnos_output"
RGB_PATH="/home/pano/hiwi/nora/FoundationPose/demo_data/$OBJECT/tmp/color.png"

rm -r /home/pano/hiwi/nora/FoundationPose/demo_data/$OBJECT/cnos_output/cnos_results

python -m utils.save_image --rgb_path $RGB_PATH

cd ../cnos
python -m src.scripts.inference_custom --template_dir $OUTPUT_DIR --rgb_path $RGB_PATH --stability_score_thresh 0.5