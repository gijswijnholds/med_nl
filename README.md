# MED-NL: Reasoning with monotonicity in Dutch Natural Language Inference

The code to go alongside our EACL 2023 paper

"Assessing Monotonicity Reasoning in Dutch through Natural Language Inference"

If you found any of the code/data/paper useful, please cite our article:

```
@inproceedings{wijnholds-2023-assessing,
    title = "Assessing Monotonicity Reasoning in {D}utch through Natural Language Inference",
    author = "Wijnholds, Gijs",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.110",
    pages = "1464--1470",
    abstract = "In this paper we investigate monotonicity reasoning in Dutch, through a novel Natural Language Inference dataset. Monotonicity reasoning shows to be highly challenging for Transformer-based language models in English and here, we corroborate those findings using a parallel Dutch dataset, obtained by translating the Monotonicity Entailment Dataset of Yanaka et al. (2019). After fine-tuning two Dutch language models BERTje and RobBERT on the Dutch NLI dataset SICK-NL, we find that performance severely drops on the monotonicity reasoning dataset, indicating poor generalization capacity of the models. We provide a detailed analysis of the test results by means of the linguistic annotations in the dataset. We find that models struggle with downward entailing contexts, and argue that this is due to a poor understanding of negation. Additionally, we find that the choice of monotonicity context affects model performance on conjunction and disjunction. We hope that this new resource paves the way for further research in generalization of neural reasoning models in Dutch, and contributes to the development of better language technology for Natural Language Inference, specifically for Dutch.",
}
```
