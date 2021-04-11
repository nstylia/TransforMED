# TransforMED: End-to-End Transformers for Evidence-Based Medicine and Argument Mining in medical literature.

This repository hosts the implementation described in our paper [TransforMED: End-to-End Transformers for Evidence-Based Medicine and Argument Mining in medical literature](https://www.sciencedirect.com/science/article/pii/S1532046421000964).
It is based on the [Transformer based Argument Mining](https://gitlab.com/tomaye/ecai2020-transformer_based_am) 
code and the [AbstRCT corpus](https://gitlab.com/tomaye/abstrct) as described in the publication 
[Transformer-based Argument Mining for Healthcare Applications](https://hal.archives-ouvertes.fr/hal-02879293/document),
 which we are extending by the following contributions. Hence, similarities to the original code and repository are expected. 

# Introduction
TransforMED is a state-of-the-art Evidence-Based Medicine (EBM) and Medical Argument Mining (MAM) system that is comprised of three models working in unison.
The system, which works in a pipeline is using EBM predictions to enhance both the Argument Identification and Argument Relation Classification models.

<img src="https://github.com/nstylia/TransforMED/blob/main/transformed_pipeline_overview_examples.png" width="800">


Our contributions are as follows:
1) State-of-the-art EBM model 
2) State-of-the-art MAM pipeline (comprised of Argument Identification and Argument Relation Classification models). 
3) Effective combination of EBM and MAM, with the first granting significant performance improvement to the second. 

# TransforMED models architecture

<img src="https://github.com/nstylia/TransforMED/blob/main/transformed_models_architectures_color.png" width="800">

# Requirements
With Python 3.6 or higher, use the requirements file provided as follows: 

```
pip install -r requirements.txt
```
 
You will also need to download, and format accordingly, the [AbstRCT corpus](https://gitlab.com/tomaye/abstrct) using the
 scripts provided in ``utils`` and ``preprocessing``. The preprocessed EBM data are provided in ``data/pico_ner/``.

# Citation
If you find our work interesting or use our work, please cite using the following:


```
@article{STYLIANOU2021103767,
title = {TransforMED: End-to-Εnd Transformers for Evidence-Based Medicine and Argument Mining in medical literature},
journal = {Journal of Biomedical Informatics},
volume = {117},
pages = {103767},
year = {2021},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2021.103767},
url = {https://www.sciencedirect.com/science/article/pii/S1532046421000964},
author = {Nikolaos Stylianou and Ioannis Vlahavas},
keywords = {Deep learning, Natural language processing, Evidence based medicine, Argument mining},
abstract = {Argument Mining (AM) refers to the task of automatically identifying arguments in a text and finding their relations. In medical literature this is done by identifying Claims and Premises and classifying their relations as either Support or Attack. Evidence-Based Medicine (EBM) refers to the task of identifying all related evidence in medical literature to allow medical practitioners to make informed choices and form accurate treatment plans. This is achieved through the automatic identification of Population, Intervention, Comparator and Outcome entities (PICO) in the literature to limit the collection to only the most relevant documents. In this work, we combine EBM with AM in medical literature to increase the performance of the individual models and create high quality argument graphs, annotated with PICO entities. To that end, we introduce a state-of-the-art EBM model, used to predict the PICO entities and two novel Argument Identification and Argument Relation classification models that utilize the PICO entities to enhance their performance. Our final system works in a pipeline and is able to identify all PICO entities in a medical publication, the arguments presented in them and their relations.}
}
```
>Stylianou, Nikolaos, and Ioannis Vlahavas. 
"[TransforMED: End-to-End Transformers for Evidence-Based Medicine and Argument Mining in medical literature](https://www.sciencedirect.com/science/article/pii/S1532046421000964)" Journal of Biomedical Informatics 117 (2021): 103767.