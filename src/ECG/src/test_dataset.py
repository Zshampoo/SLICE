# test_dataset.py

from datasets import Dataset

# example pairs: (content, style)
data = [
    {"content": "The patient is a 45-year-old male with a history of hypertension.", "style": "This individual is a 45-year-old male with a known history of hypertension."},
    {"content": "The patient complains of chest pain and shortness of breath.", "style": "This individual reports chest discomfort and difficulty breathing."},
    {"content": "The doctor suggests the patient take aspirin daily.", "style": "The physician recommends the patient to take aspirin on a daily basis."},
    {"content": "The test results indicate a mild infection.", "style": "The lab results show a mild infection."},
    {"content": "The patient was diagnosed with type 2 diabetes.", "style": "This individual has been diagnosed with type 2 diabetes."},
]

# create Dataset
dataset = Dataset.from_dict(data)

# print dataset contents
print(dataset)
