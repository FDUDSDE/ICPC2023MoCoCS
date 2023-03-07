### Dataset
The CodeSearchNet dataset we use comes from the settings of GraphCodeBERT. 
You can download and preprocess data using the following command.
```shell
unzip dataset.zip
cd dataset
bash run.sh 
cd ..
```
### Dependency
You can install dependencies using the following command.
```shell
pip install torch
pip install transformers
pip install tree_sitter
```
### Fine-tune
We fine-tuned the model on a V100-32G GPU. We provide a script for fine-tuning the model. 
You can change the programming language type and initial model in the script. 
And you can fine-tune the model using the following command.
```shell
sh run.sh
```
Supported programming languages: ruby, javascript, python, java, php and go.

Supported initial pre-trained models: unixcoder-base, codebert-base and graphcodebert-base.
### Evaluation
We also provide a shell for evaluating the fine-tuned model on the test set. 
Noted that the programming language and initial model need to be consistent with the fine-tuning script.
```shell
sh test.sh
```