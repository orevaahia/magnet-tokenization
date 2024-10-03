# Magnet-Tokenization


Paper: [MAGNET: Improving the Multilingual Fairness of Language Models with Adaptive Gradient-Based Tokenization](https://arxiv.org/abs/2407.08818)


# Environment
```
bash conda create -n magnet python=3.8
pip install -r requirements.txt
```

# Data
We sampled a portion of data for each language from the [OSCAR corpus](https://huggingface.co/datasets/oscar-corpus/oscar). You can download [here](https://drive.google.com/drive/folders/1ea_nPFUc3ga_P3hfIZGb2qFDY9H06e1Y?usp=drive_link) and place in a folder called `data/`


# Configs
The config files are located in `configs/`.  The main section to be modified here is the `boundaries` section.<br>
`script_tokens`: denotes new tokens to add to the byte-vocabulary for each script-name.<br>
`prior_list`: Direct mapping of script to the binomial prior that controls the compression rate.<br>
`temp`: temperature for Gumbel Sigmoid


# Training
```
# With Magnet:
bash scripts/run_train.sh
```

# Downstream evaluation


