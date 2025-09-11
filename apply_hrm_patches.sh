#!/bin/bash
# Apply HRM patches for RTX 4090 compatibility

echo "Applying HRM patches for RTX 4090..."

# Navigate to HRM directory
cd HRM

# Apply patches
echo "Patching requirements.txt..."
sed -i 's/adam-atan2/adam-atan2-pytorch/g' requirements.txt

echo "Patching pretrain.py..."
sed -i 's/adam_atan2/adam_atan2_pytorch/g' pretrain.py
sed -i 's/AdamATan2/AdamAtan2/g' pretrain.py
sed -i 's/lr=0,/lr=0.0001,/g' pretrain.py

echo "Patches applied successfully!"

