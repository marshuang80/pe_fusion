# PE Fusion 

This repository contains the scripts and models used in the paper *"Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record: a case-study in pulmonary embolism detection"* published on Nature Scientific Reports.[manuscript link](https://www-nature-com.stanford.idm.oclc.org/articles/s41598-020-78888-w)


## Abstract 

Recent advancements in deep learning have led to a resurgence of medical imaging and Electronic Medical Record (EMR) models for a variety of applications, including clinical decision support, automated workflow triage, clinical prediction and more. However, very few models have been developed to integrate both clinical and imaging data, despite that in routine practice clinicians rely on EMR to provide context in medical imaging interpretation. In this study, we developed and compared different multimodal fusion model architectures that are capable of utilizing both pixel data from volumetric Computed Tomography Pulmonary Angiography scans and clinical patient data from the EMR to automatically classify Pulmonary Embolism (PE) cases. The best performing multimodality model is a late fusion model that achieves an AUROC of 0.947 [95% CI: 0.946–0.948] on the entire held-out test set, outperforming imaging-only and EMR-only single modality models.

## Methods

In this study, we separately trained an imaging-only model [PENet](https://rdcu.be/b3Lll) and 7 EMR-only neural network models for each modality. These single modality models not only serve as baselines for performance comparison, they also provide different inputs and components for different fusion models. A total of 7 fusion architectures were implemented and compared:

![](./img/fusion_architectures.tiff)

## Results

Model performance on the held-out testset with 95% confidence interval using probability threshold that maximizes both sensitivity and specificity on the validation dataset. 

|                   |Imaging model      |EMR model          |Late elastic average Fusion|
|-------------------|-------------------|-------------------|---------------------------|
|Operating threshold|0.625              |0.63               |0.448                      |
|Accuracy           |0.687 [0.685–0.689]|0.834 [0.832–0.835]|0.885 [0.884–0.886]        |
|AUROC              |0.791 [0.788–0.793]|0.911 [0.910–0.913]|0.947 [0.946–0.948]        |
|Specificity        |0.862 [0.860–0.865]|0.875 [0.872–0.877]|0.902 [0.9–0.904]          |
|Sensitivity        |0.559 [0.557–0.562]|0.804 [0.801–0.806]|0.873 [0.871–0.875]        |
|PPV                |0.848 [0.846–0.851]|0.898 [0.896–0.899]|0.924 [0.923–0.926         |
|NPV                |0.588 [0.585–0.590]|0.765 [0.761–0.767]|0.838 [0.835–0.84]         |


## Citation
Huang, SC., Pareek, A., Zamanian, R. et al. Multimodal fusion with deep neural networks for leveraging CT imaging and electronic health record: a case-study in pulmonary embolism detection. Sci Rep 10, 22147 (2020). https://doi-org.stanford.idm.oclc.org/10.1038/s41598-020-78888-w

## Data Availability 
The datasets generated and analyzed during the study are not currently publicly available due to HIPAA compliance agreement but are available from the corresponding author on reasonable request.