# Exploring Medical Text Summarization with John Snow Labs

## ➤ Introduction

Text Summarization, a crucial task in Natural Language Processing (NLP), condenses lengthy text documents into shorter, compact versions while retaining the most critical information and meaning. The primary goal is to create a summary that accurately echoes the content of the original text in a more succinct form.

In the context of healthcare, where the prompt and accurate comprehension of extensive medical literature is paramount, medical text summarization plays a significant role.

## ➤ Purpose 

John Snow Labs offers a suite of specialized text summarization models tailored for the healthcare industry. This joint Solution Accelerator serves as a definitive guide for utilizing these models via Spark NLP for Healthcare libraries, offering users an opportunity to harness the power of NLP for summarization tasks.

🪄 MODELS:
|   Model Name                            |   Model Description                                                                                                                                                       |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   [summarizer_clinical_jsl](https://nlp.johnsnowlabs.com/2023/03/25/summarizer_clinical_jsl.html)                |   This summarization model can quickly summarize clinical notes, encounters, critical care notes, discharge notes, reports, etc. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                                   |
|   [summarizer_clinical_jsl_augmented](https://nlp.johnsnowlabs.com/2023/03/30/summarizer_clinical_jsl_augmented_en.html)     |   This summarization model can quickly summarize clinical notes, encounters, critical care notes, discharge notes, reports, etc. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. This model is further optimized by augmenting the training methodology, and dataset. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                                    |
|   [summarizer_biomedical_pubmed](https://nlp.johnsnowlabs.com/2023/04/03/summarizer_biomedical_pubmed_en.html)          |   This summarization model can quickly summarize biomedical research or short papers. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                        |
|   [summarizer_generic_jsl](https://nlp.johnsnowlabs.com/2023/03/30/summarizer_generic_jsl_en.html)                |   A model optimized with custom data and training methodology from John Snow Labs, generating summaries from clinical notes.                                              |
|   [summarizer_clinical_questions](https://nlp.johnsnowlabs.com/2023/04/03/summarizer_clinical_questions_en.html)         |   This summarization model efficiently summarizes medical questions from a range of clinical sources, providing concise summaries and generating relevant questions. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                        |
|   [summarizer_radiology](https://nlp.johnsnowlabs.com/2023/04/23/summarizer_jsl_radiology_en.html)                  |   This summarization model enables users to rapidly access a succinct synopsis of a report’s key findings without compromising on essential details. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                            |
|   [summarizer_clinical_guidelines_large](https://nlp.johnsnowlabs.com/2023/05/08/summarizer_clinical_guidelines_large_en.html)  |   The Medical Summarizer Model efficiently categorizes clinical guidelines into four sections for easy understanding. With a 768-token context length, it provides detailed yet concise summaries, leveraging data curated by doctors from John Snow Labs. |
|   [summarizer_clinical_laymen](https://nlp.johnsnowlabs.com/2023/05/31/summarizer_clinical_laymen_en.html)            |   This model has been carefully fine-tuned with a custom dataset curated by John Snow Labs, expressly designed to minimize the use of clinical terminology in the generated summaries. The summarizer_clinical_laymen model is capable of producing summaries of up to 512 tokens from an input text of a maximum of 1024 tokens.  |



## ➤ License

Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.

|Library Name|Library License|Library License URL|Library Source URL|
| :-: | :-:| :-: | :-:|
|Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
|Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
|NLTK | Apache License 2.0 | https://github.com/nltk/nltk/blob/develop/LICENSE.txt | https://github.com/nltk/nltk/tree/develop |
|textwrap | MIT License | https://github.com/mgeisler/textwrap/blob/master/LICENSE | https://github.com/mgeisler/textwrap/tree/master |
|Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
|Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
|Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|



|Author|
|-|
|Databricks Inc.|
|John Snow Labs Inc.|


## ➤ Disclaimers
Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.


## ➤ Instruction
To run this accelerator, set up JSL Partner Connect [AWS](https://docs.databricks.com/integrations/ml/john-snow-labs.html#connect-to-john-snow-labs-using-partner-connect), [Azure](https://learn.microsoft.com/en-us/azure/databricks/integrations/ml/john-snow-labs#--connect-to-john-snow-labs-using-partner-connect) and navigate to **My Subscriptions** tab. Make sure you have a valid subscription for the workspace you clone this repo into, then **install on cluster** as shown in the screenshot below, with the default options. You will receive an email from JSL when the installation completes.

<br>
<img src="https://raw.githubusercontent.com/databricks-industry-solutions/oncology/main/images/JSL_partner_connect_install.png" width=65%>

Once the JSL installation completes successfully, clone this repo into a Databricks workspace. Attach the `RUNME` notebook to any cluster and execute the notebook via `Run-All`. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.
