Zu AI Bot
Description
The Zu AI Bot is an NLP project aimed at developing a chatbot or text generation system. This project utilizes deep learning techniques, leveraging PyTorch to implement encoder-decoder models. It supports Chinese text tokenization with jieba, tracks experiments via Weights & Biases (wandb), and evaluates model performance using BLEU scores.

Installation
Dependencies
The project requires the following libraries:

Python 3.x
PyTorch
Jieba
TorchText
Scikit-learn
Numpy
Matplotlib
tqdm
Weights & Biases (wandb)
You can install these dependencies via pip:

`pip install torch jieba torchtext scikit-learn numpy matplotlib tqdm wandb`

Additional Tools
Weights & Biases: Used for experiment tracking. You'll need to set up your account:

Install Weights & Biases: pip install wandb
Login: wandb login
Initialize a project: wandb.init(project='Zu_bot', entity='ZU')
Dataset: Ensure that you have access to the necessary dataset, and format it according to your needs (e.g., tabular CSV format). Modify paths if needed.

Usage
1. Data Preparation
Ensure the dataset is split into training, validation, and test sets according to the specified ratio in the config (70% training, 20% validation, 10% test). Use the relevant sections of the script to load and preprocess your data.

2. Experiment Tracking
Weights & Biases (wandb) is integrated to track training metrics and results. All logged metrics (e.g., loss, BLEU scores) will be available in your WandB dashboard for real-time monitoring.
Configuration
The configuration parameters for the model are defined in the script and can be adjusted:

split_ratio: [0.7, 0.2, 0.1] – Dataset split ratio for training, validation, and testing
batch_size: 185 – Batch size for training
embedding_dim: 256 – Word embedding dimension
nhead: 8 – Number of attention heads in the transformer model
num_encoder_layers: 5 – Number of encoder layers in the transformer
num_decoder_layers: 5 – Number of decoder layers in the transformer
Feel free to modify these parameters to better suit your dataset and hardware configuration.

Training
The training process involves the following steps:

Load Dataset: The dataset is tokenized and split into batches.
Train Model: The encoder-decoder model is trained on the dataset with backpropagation.
Track Metrics: Training progress is tracked via loss, BLEU scores, etc.
Visualizations: Visualizations like loss curves can be generated using matplotlib.
Evaluation
The model's performance is evaluated using BLEU scores, a standard metric for assessing the quality of text generation or translation tasks. The evaluation script within the model provides these scores at the end of each epoch.

`BLEU Score: 0.xx  # Example output`

License
This project is open-sourced under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.

Contact Information
For any questions, feel free to contact the project maintainer:

Name: deviser
Email: yaoyuanou@gmail.com
