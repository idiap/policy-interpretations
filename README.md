# policy-interpretations
This repository contains the code and links to the data and trained models for the paper [Generating Interpretations of Policy Announcements](https://aclanthology.org/2024.nlp4dh-1.50/) presented at the NLP4DH workshop at EMNLP 2024.

## Contents
1. [Short Description](#short-description)
2. [Data](#data)
3. [Models](#models)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contact](#contact)
7. [Authors and Acknowledgments](#authors-and-acknowledgments)
8. [Citation](#citation)

## Short Description
We present a dataset for the interpretation of statements of political actors. The dataset consists of policy announcements and corresponding annotated interpretations, on the topic of US foreign policy relations with Russia in the years 1993 up to 2016. We analyze the performance of finetuning standard sequence-to-sequence models of varying sizes on predicting the annotated interpretations and compare them to few-shot prompted large language models. We find that 1) model size is not the main factor for success on this task, 2) finetuning smaller models provides both quantitatively and qualitatively superior results to in-context learning with large language models, but 3) large language models pick up the annotation format and approximate the category distribution with just a few in-context examples.

## Data
The dataset can be accessed with different preprocessing applied:
- [Text dataset](https://drive.switch.ch/index.php/s/Y7mCJHUie31EDr0): The source and target files in plain text format.
- [Filtered sources](https://drive.switch.ch/index.php/s/wiHN2QpBXY3wbki): The dataset is split into train/valid/test and sources are oracle-filtered to a max length of 1024 tokens with the BART tokenizer.
- [Encoded for BART](https://drive.switch.ch/index.php/s/ot4uqkDvNVLJ2As): Encoded PyTorch dataset for BART.
- [Encoded for T5](https://drive.switch.ch/index.php/s/PNW1azeLVGA5JHm): Encoded PyTorch dataset for T5/Flan-T5 models.

The data is shared under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Models
We provide the [model checkpoint](https://drive.switch.ch/index.php/s/HjPRcDt0u56lbQ9) of our best-performing finetuned BART model.

The models are shared under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Installation
First, install conda, e.g. from [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Then create and activate the environment:
```
conda env create -f environment.yml
conda activate policy-interpretations
```

## Usage
### Finetuning
To train a model, use the [main.py](main.py) script. The default arguments are set to the hyperparameter values we used in our experiments. Here's an example of how to train BART with the parameters we used ourselves:
```
python main.py \
--model bart \
--data_dir data_us_russia_bart \
--model_dir models/bart \
--default_root_dir logs/bart \
--deterministic \
--gpus 1 \
--batch_size 2 \
--accumulate_grad_batches 2 \
--max_epochs 20 \
--min_epochs 10 \
--max_steps 16800
```

### Few-shot Prompting
The [few_shot.py](few_shot.py) script implements few-shot prompting an LLM. As opposed to all other scripts, it requires the dependencies and library versions in [environment_few_shot.yml](environment_few_shot.yml).

### Saving Model Outputs
To run evaluations for sequence-to-sequence models, you have to first save their outputs in text format. Run [save_model_outputs.py](save_model_outputs.py) with the default parameters. Don't forget to specify `model_dir` and `output_dir`. 

### Evaluation
Use the [evaluations.py](evaluations.py) script to run the text generation evaluations and specify the path to your model outputs as the `input_dir`. Results will be saved as a json file in the same directory.

## Contact
In case of problems or questions open a Github issue or write an email to andreas.marfurt [at] hslu.ch.

## Authors and Acknowledgments
This paper was written by Andreas Marfurt, Ashley Thornton, David Sylvan, and James Henderson.

The work was supported as a part of the grant Automated interpretation of political and economic policy documents: Machine learning using semantic and syntactic information, funded by the Swiss National Science Foundation (grant number CRSII5_180320), and led by the co-PIs James Henderson, Jean-Louis Arcand and David Sylvan.

## Citation
If you use our code, data or models, please cite us.
```
@inproceedings{marfurt-etal-2024-generating,
    title = "Generating Interpretations of Policy Announcements",
    author = "Marfurt, Andreas  and
      Thornton, Ashley  and
      Sylvan, David  and
      Henderson, James",
    editor = {H{\"a}m{\"a}l{\"a}inen, Mika  and
      {\"O}hman, Emily  and
      Miyagawa, So  and
      Alnajjar, Khalid  and
      Bizzoni, Yuri},
    booktitle = "Proceedings of the 4th International Conference on Natural Language Processing for Digital Humanities",
    month = nov,
    year = "2024",
    address = "Miami, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.nlp4dh-1.50",
    pages = "513--520",
}

```
