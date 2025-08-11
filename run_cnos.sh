OBJECT="T"
OUTPUT_DIR="/home/pano/hiwi/nora/FoundationPose/demo_data/$OBJECT/new_render"
RGB_PATH=/home/pano/Datasets/pushT/2025_07_21-10_50_21/images/ORB_0_orig/0.png

rm -r /home/pano/hiwi/nora/FoundationPose/demo_data/$OBJECT/new_render/cnos_results

python -m utils.save_image --rgb_path $RGB_PATH

cd ../cnos
python -m src.scripts.inference_custom --template_dir $OUTPUT_DIR --rgb_path $RGB_PATH --stability_score_thresh 0.5