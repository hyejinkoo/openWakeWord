# Generate clips
# echo "step1. Generate clips"
# python /Users/koo/dev/openwakeword/openwakeword/train.py --training_config config.yaml --generate_clips

# Step 2: Augment the generated clips
echo "step 2. Augment the generated clips"
python /Users/koo/dev/openwakeword/openwakeword/train.py --training_config config.yaml --augment_clips

# Step 3: Train model
echo "step 3: Train model"
python /Users/koo/dev/openwakeword/openwakeword/train.py --training_config config.yaml --train_model

# Step 4: Convert onnx model to tflite
echo "step 4: Convert onnx model to tflite"
python onnx_to_tflite.py

echo "Done"