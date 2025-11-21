# Multi-Stage Job Advertisement Analysis using BERT and LLMs

**Team:**  
- Mikaël Bonvin
- Govender Vimerlin  
- Mouhamadou Thiam

---

## Project Overview

This project develops a two-stage NLP pipeline with an extended mutli agent orchestrator: 
1. **Zone Identification:** Fine-tune a multilingual BERT model to classify text sections in job advertisements.
2. **Skill Extraction:** Use Google Gemini to extract professional skills from the “Skills and Content” zone.
3. **Multi-agent orchestration system:** Coordinate multiple LLMs for improved skill extraction robustness.

---

## Repository Structure
```
├── data  
│   └── annotated.json (untracked in Git - available in Google Drive) 
│
├── notebooks  
│   ├── 01_data_preparation.ipynb  
│   ├── 02_train_bert.ipynb  
│   ├── 03_evaluation.ipynb  
│   ├── 04_skill_extraction_llm.ipynb  
│   └── 05_agent_orchestration_optional.ipynb
│
├── src  
│   ├── preprocessing.py  
│   └── training_template.py  
│
├── README.md  
└── requirements.txt  
```

Each notebook is runnable in **Google Colab** — simply click "Open in Colab" badge:

---

## Requirements

Install dependencies locally or in Colab:

```bash
pip install -r requirements.txt
