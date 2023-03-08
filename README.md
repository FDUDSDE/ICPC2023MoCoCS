### Dataset
The CodeSearchNet dataset we use comes from the settings of [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch). 
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
### Cite
```shell
@inproceedings{shi2022mococs,
  title={Improving Code Search with Multi-Modal Momentum Contrastive Learning},
  author={Shi, Zejian and Xiong, Yun and Zhang, Yao and Jiang, Zhijie and Zhao, Jinjing and Wang, Lei and Li, Shanshan},
  booktitle={Proceedings of the 31st IEEE/ACM International Conference on Program Comprehension},
  year={2023}
  organization={IEEE/ACM}
}
```
### Acknowledgement
We use the parser in [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch) to convert code into AST and DFG. 

