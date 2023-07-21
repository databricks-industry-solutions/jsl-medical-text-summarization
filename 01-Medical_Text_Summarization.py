# Databricks notebook source
# MAGIC %md
# MAGIC ![JohnSnowLabs](https://nlp.johnsnowlabs.com/assets/images/logo.png)

# COMMAND ----------

# MAGIC %md
# MAGIC # 🪄 Medical Text Summarization
# MAGIC
# MAGIC 🔎 Text Summarization is a natural language processing (NLP) task that involves condensing a lengthy text document into a shorter, more compact version while still retaining the most important information and meaning. The goal is to produce a summary that accurately represents the content of the original text in a concise form.
# MAGIC
# MAGIC 🔎There are `different approaches` to text summarization, including `extractive methods that` identify and extract important sentences or phrases from the text, and `abstractive methods` that generate new text based on the content of the original text.

# COMMAND ----------

import os
import json
import string
import numpy as np
import pandas as pd

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.util import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *
from sparknlp.pretrained import ResourceDownloader
from sparknlp.pretrained import  PretrainedPipeline

from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.expand_frame_repr', False)
import textwrap

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ![IMAGE](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/data/Automated_Summarization_Clinical_Notes.png?raw=true)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ![IMAGE](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/data/Summarization_Methods_vs_Quality_Dimensions.png?raw=true)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔎 Models

# COMMAND ----------

# MAGIC %md
# MAGIC |   Model Name                            |   Model Description                                                                                                                                                       |
# MAGIC |-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC |   [summarizer_clinical_jsl](https://nlp.johnsnowlabs.com/2023/03/25/summarizer_clinical_jsl.html)                |   This summarization model can quickly summarize clinical notes, encounters, critical care notes, discharge notes, reports, etc. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                                   |
# MAGIC |   [summarizer_clinical_jsl_augmented](https://nlp.johnsnowlabs.com/2023/03/30/summarizer_clinical_jsl_augmented_en.html)     |   This summarization model can quickly summarize clinical notes, encounters, critical care notes, discharge notes, reports, etc. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. This model is further optimized by augmenting the training methodology, and dataset. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                                    |
# MAGIC |   [summarizer_biomedical_pubmed](https://nlp.johnsnowlabs.com/2023/04/03/summarizer_biomedical_pubmed_en.html)          |   This summarization model can quickly summarize biomedical research or short papers. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                        |
# MAGIC |   [summarizer_generic_jsl](https://nlp.johnsnowlabs.com/2023/03/30/summarizer_generic_jsl_en.html)                |   A model optimized with custom data and training methodology from John Snow Labs, generating summaries from clinical notes.                                              |
# MAGIC |   [summarizer_clinical_questions](https://nlp.johnsnowlabs.com/2023/04/03/summarizer_clinical_questions_en.html)         |   This summarization model efficiently summarizes medical questions from a range of clinical sources, providing concise summaries and generating relevant questions. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                        |
# MAGIC |   [summarizer_radiology](https://nlp.johnsnowlabs.com/2023/04/23/summarizer_jsl_radiology_en.html)                  |   This summarization model enables users to rapidly access a succinct synopsis of a report’s key findings without compromising on essential details. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).                                                            |
# MAGIC |   [summarizer_clinical_guidelines_large](https://nlp.johnsnowlabs.com/2023/05/08/summarizer_clinical_guidelines_large_en.html)  |   The Medical Summarizer Model efficiently categorizes clinical guidelines into four sections for easy understanding. With a 768-token context length, it provides detailed yet concise summaries, leveraging data curated by doctors from John Snow Labs. |
# MAGIC |   [summarizer_clinical_laymen](https://nlp.johnsnowlabs.com/2023/05/31/summarizer_clinical_laymen_en.html)            |   This model has been carefully fine-tuned with a custom dataset curated by John Snow Labs, expressly designed to minimize the use of clinical terminology in the generated summaries. The summarizer_clinical_laymen model is capable of producing summaries of up to 512 tokens from an input text of a maximum of 1024 tokens.  |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📍Benchmark Report
# MAGIC
# MAGIC Our clinical summarizer models with only 250M parameters perform 30-35% better than non-clinical SOTA text summarizers with 500M parameters, in terms of Bleu and Rouge benchmarks. That is, we achieve 30% better with half of the parameters that other LLMs have. See the details below.
# MAGIC
# MAGIC
# MAGIC ### 🔎Benchmark on Samsum Dataset
# MAGIC
# MAGIC | model_name | model_size | rouge | bleu | bertscore_precision | bertscore_recall: | bertscore_f1 |
# MAGIC |--|--|--|--|--|--|--|
# MAGIC philschmid/flan-t5-base-samsum | 240M | 0.2734 | 0.1813 | 0.8938 | 0.9133 | 0.9034 | 
# MAGIC linydub/bart-large-samsum | 500M | 0.3060 | 0.2168 | 0.8961 | 0.9065 | 0.9013 |
# MAGIC philschmid/bart-large-cnn-samsum | 500M | 0.3794 | 0.1262 | 0.8599 | 0.9153 | 0.8867 | 
# MAGIC transformersbook/pegasus-samsum | 570M | 0.3049 | 0.1543 | 0.8942 | 0.9183 | 0.9061 | 
# MAGIC summarizer_generic_jsl | 240M | 0.2703 | 0.1932 | 0.8944 | 0.9161 | 0.9051 |
# MAGIC
# MAGIC
# MAGIC ### 🔎Benchmark on MtSamples Summarization Dataset
# MAGIC
# MAGIC | model_name | model_size | rouge | bleu | bertscore_precision | bertscore_recall: | bertscore_f1 |
# MAGIC |--|--|--|--|--|--|--|
# MAGIC philschmid/flan-t5-base-samsum | 250M | 0.1919 | 0.1124 | 0.8409 | 0.8964 | 0.8678 | 
# MAGIC linydub/bart-large-samsum | 500M | 0.1586 | 0.0732 | 0.8747 | 0.8184 | 0.8456 | 
# MAGIC philschmid/bart-large-cnn-samsum |  500M | 0.2170 | 0.1299 | 0.8846 | 0.8436 | 0.8636 |
# MAGIC transformersbook/pegasus-samsum | 500M | 0.1924 | 0.0965 | 0.8920 | 0.8149 | 0.8517 | 
# MAGIC summarizer_clinical_jsl | 250M | 0.4836 | 0.4188 | 0.9041 | 0.9374 | 0.9204 | 
# MAGIC summarizer_clinical_jsl_augmented | 250M | 0.5119 | 0.4545 | 0.9282 | 0.9526 | 0.9402 |
# MAGIC
# MAGIC
# MAGIC ### 🔎Benchmark on MIMIC Summarization Dataset
# MAGIC
# MAGIC | model_name | model_size | rouge | bleu | bertscore_precision | bertscore_recall: | bertscore_f1 |
# MAGIC |--|--|--|--|--|--|--|
# MAGIC philschmid/flan-t5-base-samsum | 250M | 0.1910 | 0.1037 | 0.8708 | 0.9056 | 0.8879 | 
# MAGIC linydub/bart-large-samsum | 500M | 0.1252 | 0.0382 | 0.8933 | 0.8440 | 0.8679 |
# MAGIC philschmid/bart-large-cnn-samsum | 500M | 0.1795 | 0.0889 | 0.9172 | 0.8978 | 0.9074 | 
# MAGIC transformersbook/pegasus-samsum | 570M | 0.1425 | 0.0582 | 0.9171 | 0.8682 | 0.8920 |
# MAGIC summarizer_clinical_jsl | 250M | 0.395 | 0.2962 | 0.895 | 0.9316 | 0.913 | 
# MAGIC summarizer_clinical_jsl_augmented | 250M | 0.3964 | 0.307 | 0.9109 | 0.9452 | 0.9227 |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📃 summarizer_clinical_jsl
# MAGIC
# MAGIC This summarization model can quickly summarize clinical notes, encounters, critical care notes, discharge notes, reports, etc. It is based on a large language model that is finetuned with healthcare-specific additional data curated by medical doctors at John Snow Labs. It can generate summaries up to 512 tokens given an input text (max 1024 tokens).

# COMMAND ----------

text = """ Patient with hypertension, syncope, and spinal stenosis - for recheck.
 (Medical Transcription Sample Report)
 SUBJECTIVE:
 The patient is a 78-year-old female who returns for recheck. She has hypertension. She denies difficulty with chest pain, palpations, orthopnea, nocturnal dyspnea, or edema.
 PAST MEDICAL HISTORY / SURGERY / HOSPITALIZATIONS:
 Reviewed and unchanged from the dictation on 12/03/2003.
 MEDICATIONS:
 Atenolol 50 mg daily, Premarin 0.625 mg daily, calcium with vitamin D two to three pills daily, multivitamin daily, aspirin as needed, and TriViFlor 25 mg two pills daily. She also has Elocon cream 0.1% and Synalar cream 0.01% that she uses as needed for rash."""


data = spark.createDataFrame([[text]]).toDF("text")
data.show(truncate = 60)

# COMMAND ----------

document_assembler = DocumentAssembler()\
            .setInputCol('text')\
            .setOutputCol('document')

summarizer = MedicalSummarizer.pretrained("summarizer_clinical_jsl", "en", "clinical/models")\
            .setInputCols(['document'])\
            .setOutputCol('summary')\
            .setMaxTextLength(512)\
            .setMaxNewTokens(512)

pipeline = Pipeline(stages=[
            document_assembler,
            summarizer])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

result = model.transform(data)

result.show()

# COMMAND ----------

result.select("summary.result").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📍 LightPipelines

# COMMAND ----------

text = """The patient is a pleasant 17-year-old gentleman who was playing basketball today in gym. Two hours prior to presentation, he started to fall and someone stepped on his ankle and kind of twisted his right ankle and he cannot bear weight on it now. It hurts to move or bear weight. No other injuries noted. He does not think he has had injuries to his ankle in the past.
SOCIAL HISTORY: He does not drink or smoke.
MEDICAL DECISION MAKING:
He had an x-ray of his ankle that showed a small ossicle versus avulsion fracture of the talonavicular joint on the lateral view. He has had no pain over the metatarsals themselves. This may be a fracture based upon his exam. He does want to have me to put him in a splint. He was given Motrin here. He will be discharged home to follow up with Dr. X from Orthopedics.
DISPOSITION: Crutches and splint were administered here. I gave him a prescription for Motrin and some Darvocet if he needs to length his sleep and if he has continued pain to follow up with Dr. X. Return if any worsening problems."""

light_model = LightPipeline(model)
light_result = light_model.annotate(text)

print("➤ DOCUMENT: \n", light_result["document"][0])
print("\n")
wrapped_text = textwrap.fill(light_result["summary"][0], width=120)
print("➤ SUMMARY: \n{}".format(wrapped_text))
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📍 Summarization of Long Documents

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚩 summaries from paragraphs in text

# COMMAND ----------

document_assembler = DocumentAssembler()\
            .setInputCol('text')\
            .setOutputCol('document')

sentenceDetector = SentenceDetectorDLModel\
            .pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
            .setInputCols(["document"])\
            .setOutputCol("sentence")\
            .setCustomBounds(["\n"])\
            .setUseCustomBoundsOnly(True)

summarizer = MedicalSummarizer\
            .pretrained("summarizer_clinical_jsl")\
            .setInputCols(['sentence'])\
            .setOutputCol('summary')\
            .setMaxTextLength(512)\
            .setMaxNewTokens(512)

pipeline = Pipeline(stages=[
            document_assembler,
            sentenceDetector,
            summarizer])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

# COMMAND ----------

text = """PRESENT ILLNESS: The patient is a 28-year-old, who is status post gastric bypass surgery nearly one year ago. He has lost about 200 pounds and was otherwise doing well until yesterday evening around 7:00-8:00 when he developed nausea and right upper quadrant pain, which apparently wrapped around toward his right side and back. He feels like he was on it but has not done so. He has overall malaise and a low-grade temperature of 100.3. He denies any prior similar or lesser symptoms. His last normal bowel movement was yesterday. He denies any outright chills or blood per rectum.

PHYSICAL EXAMINATION: His temperature is 100.3, blood pressure 129/59, respirations 16, heart rate 84. He is drowsy, but easily arousable and appropriate with conversation. He is oriented to person, place, and situation. He is normocephalic, atraumatic. His sclerae are anicteric. His mucous membranes are somewhat tacky. His neck is supple and symmetric. His respirations are unlabored and clear. He has a regular rate and rhythm. His abdomen is soft. He has diffuse right upper quadrant tenderness, worse focally, but no rebound or guarding. He otherwise has no organomegaly, masses, or abdominal hernias evident. His extremities are symmetrical with no edema. His posterior tibial pulses are palpable and symmetric. He is grossly nonfocal neurologically.

PLAN: He will be admitted and placed on IV antibiotics. We will get an ultrasound this morning. He will need his gallbladder out, probably with intraoperative cholangiogram. Hopefully, the stone will pass this way. Due to his anatomy, an ERCP would prove quite difficult if not impossible unless laparoscopic assisted. Dr. X will see him later this morning and discuss the plan further. The patient understands."""

light_model = LightPipeline(model)
light_result = light_model.annotate(text)

# COMMAND ----------

for i in range(len(light_result['sentence'])):
    document_text = textwrap.fill(light_result['sentence'][i], width=120)
    summary_text = textwrap.fill(light_result['summary'][i], width=120)

    print("➤ Document {}: \n{}".format(i+1, document_text))
    print("\n")
    print("➤ Summary {}: \n{}".format(i+1, summary_text))
    print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚩 setRefineSummary

# COMMAND ----------

# MAGIC %md
# MAGIC **We've Made Significant Enhancements To Our Text Summarization Method, Which Now Utilizes A Map-Reduce Approach For Section-Wise Summarization.**
# MAGIC
# MAGIC We are excited to announce the integration of new parameters into our `MedicalSummarizer` annotator, empowering users to overcome token limitations and attain heightened flexibility in their medical summarization endeavors. These advanced parameters significantly augment the annotator's functionality, enabling users to generate more accurate and comprehensive summaries of medical documents. By employing a map-reduce approach, the `MedicalSummarizer` efficiently condenses distinct text segments until the desired length is achieved.
# MAGIC
# MAGIC The following parameters have been introduced:
# MAGIC
# MAGIC - `setRefineSummary`: Set to True for refined summarization with increased computational cost.
# MAGIC - `setRefineSummaryTargetLength`: Define the target length of summarizations in tokens (delimited by whitespace). Effective only when setRefineSummary=True.
# MAGIC - `setRefineChunkSize`: Specify the desired size of refined chunks. Should correspond to the LLM context window size in tokens. Effective only when - `setRefineSummary=True`.
# MAGIC - `setRefineMaxAttempts`: Determine the number of attempts for re-summarizing chunks exceeding the setRefineSummaryTargetLength before discontinuing. Effective only when `setRefineSummary=True`.
# MAGIC
# MAGIC These enhancements to the MedicalSummarizer annotator represent our ongoing commitment to providing state-of-the-art tools for healthcare professionals and researchers, facilitating more efficient and accurate medical text analysis.

# COMMAND ----------

document_assembler = DocumentAssembler()\
            .setInputCol('text')\
            .setOutputCol('document')

summarizer = MedicalSummarizer.pretrained("summarizer_clinical_jsl", "en", "clinical/models")\
            .setInputCols(["document"])\
            .setOutputCol("summary")\
            .setMaxTextLength(512)\
            .setMaxNewTokens(512)\
            .setDoSample(True)\
            .setRefineSummary(True)\
            .setRefineSummaryTargetLength(100)\
            .setRefineMaxAttempts(3)\
            .setRefineChunkSize(512)\

pipeline = Pipeline(stages=[
            document_assembler,
            summarizer])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

# COMMAND ----------

text = """The patient is a pleasant 17-year-old gentleman who was playing basketball today in gym. Two hours prior to presentation, he started to fall and someone stepped on his ankle and kind of twisted his right ankle and he cannot bear weight on it now. It hurts to move or bear weight. No other injuries noted. He does not think he has had injuries to his ankle in the past.
SOCIAL HISTORY: He does not drink or smoke.
MEDICAL DECISION MAKING:
He had an x-ray of his ankle that showed a small ossicle versus avulsion fracture of the talonavicular joint on the lateral view. He has had no pain over the metatarsals themselves. This may be a fracture based upon his exam. He does want to have me to put him in a splint. He was given Motrin here. He will be discharged home to follow up with Dr. X from Orthopedics.
DISPOSITION: Crutches and splint were administered here. I gave him a prescription for Motrin and some Darvocet if he needs to length his sleep and if he has continued pain to follow up with Dr. X. Return if any worsening problems."""

light_model = LightPipeline(model)
light_result = light_model.annotate(text)

# COMMAND ----------

light_result["summary"]

# COMMAND ----------

for i in range(len(light_result['document'])):
    document_text = textwrap.fill(light_result['document'][i], width=120)
    summary_text = textwrap.fill(light_result['summary'][i], width=120)

    print("➤ Document {}: \n{}".format(i+1, document_text))
    print("\n")
    print("➤ Summary {}: \n{}".format(i+1, summary_text))
    print("\n")

# COMMAND ----------

text = """To determine whether a course of low-dose indomethacin therapy, when initiated within 24 hours of birth, would decrease ductal shunting in premature infants who received prophylactic surfactant in the delivery room. Ninety infants, with birth weights of 600 to 1250 gm, were entered into a prospective, randomized, controlled trial to receive either indomethacin, 0.1 mg/kg per dose, or placebo less than 24 hours and again every 24 hours for six doses. Echocardiography was performed on day 1 before treatment and on day 7, 24 hours after treatment. A hemodynamically significant patent ductus arteriosus (PDA) was confirmed with an out-of-study echocardiogram, and the nonresponders were treated with standard indomethacin or ligation. Forty-three infants received indomethacin (birth weight, 915 +/- 209 gm; gestational age, 26.4 +/- 1.6 weeks; 25 boys), and 47 received placebo (birth weight, 879 +/- 202 gm; gestational age, 26.4 +/- 1.8 weeks; 22 boys) (P = not significant). Of 90 infants, 77 (86%) had a PDA by echocardiogram on the first day of life before study treatment; 84% of these PDAs were moderate or large in size in the indomethacin-treated group compared with 93% in the placebo group. Nine of forty indomethacin-treated infants (21%) were study-dose nonresponders compared with 22 (47%) of 47 placebo-treated infants (p < 0.018). There were no significant differences between both groups in any of the long-term outcome variables, including intraventricular hemorrhage, duration of oxygen therapy, endotracheal intubation, duration of stay in neonatal intensive care unit, time to regain birth weight or reach full caloric intake, incidence of bronchopulmonary dysplasia, and survival. No significant differences were noted in the incidence of oliguria, elevated plasma creatinine concentration, thrombocytopenia, pulmonary hemorrhage, or necrotizing enterocolitis. The prophylactic use of low doses of indomethacin, when initiated in the first 24 hours of life in low birth weight infants who receive prophylactic surfactant in the delivery room, decreases the incidence of left-to-right shunting at the level of the ductus arteriosus."""

light_result = light_model.annotate(text)

# COMMAND ----------

light_result["summary"]

# COMMAND ----------

# MAGIC %md
# MAGIC # License
# MAGIC Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC |Library Name|Library License|Library License URL|Library Source URL|
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Pandas |BSD 3-Clause License| https://github.com/pandas-dev/pandas/blob/master/LICENSE | https://github.com/pandas-dev/pandas|
# MAGIC |Numpy |BSD 3-Clause License| https://github.com/numpy/numpy/blob/main/LICENSE.txt | https://github.com/numpy/numpy|
# MAGIC |NLTK | Apache License 2.0 | https://github.com/nltk/nltk/blob/develop/LICENSE.txt | https://github.com/nltk/nltk/tree/develop |
# MAGIC |textwrap | MIT License | https://github.com/mgeisler/textwrap/blob/master/LICENSE | https://github.com/mgeisler/textwrap/tree/master |
# MAGIC |Apache Spark |Apache License 2.0| https://github.com/apache/spark/blob/master/LICENSE | https://github.com/apache/spark/tree/master/python/pyspark|
# MAGIC |Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
# MAGIC |Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
# MAGIC
# MAGIC
# MAGIC
# MAGIC |Author|
# MAGIC |-|
# MAGIC |Databricks Inc.|
# MAGIC |John Snow Labs Inc.|

# COMMAND ----------

# MAGIC %md
# MAGIC # Disclaimers
# MAGIC Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.
