# NIH ChestX-ray14 Classifier
This project trains a deep learning model to classify chest X-rays using the NIH ChestX-ray14 dataset.

## Features
- Multi-label disease classification
- PyTorch training pipeline

## How to Use
1. Download data from Kaggle
2. Place images in the folder`data/`
3. Split dataset into training+validation and testing sets:

python utils/split_dataset.py

4. Modify image paths in the input csv file:

python utils/change_path.py

4. Run training:

python src/train.py

5. Run evaluation:

python src/eval.py

## License
This project is licensed under the MIT License.