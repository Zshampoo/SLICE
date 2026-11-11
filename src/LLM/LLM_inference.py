import json
import os
import re
from pathlib import Path
from typing import Optional, List
from openai import AsyncOpenAI
import pandas as pd
import asyncio
import ast
import openpyxl
import time
from deep_translator import GoogleTranslator

# from src.LLM.CE_calc import clean_report_mimic_cxr


class Refined_Report:
    def __init__(
            self,
            api_key: str,
            base_url: str = 'https://api.openai.com/v1',
            engine: str = "gpt-4.1-mini",
            temperature: float = 0.1,
            max_tokens: int = 512,
            type = 'PRED'
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        # self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.type = type

    # def _build_prompt(self, global_abnormalities: str, local_abnormalities: str, report: str) -> str:
    #     return f"""
    # You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image model.

    # Optimization depends on:
    # 1. The abnormal types identified by the image classifier. A value of (type, 0) means no abnormal type, and (type, 1) means it has an abnormal type.

    # 2. The normal/abnormal status of 27 anatomical regions from the image model output. Regions not mentioned should be considered normal.

    # Optimization goals:
    # - Integrate the information provided by the image model (abnormal types and anatomical region statuses).
    # - Utilize professional knowledge (such as causes and morphological features of abnormalities in specific regions).
    # - Ensure the optimized report maintains the style and professionalism of the original report, while being more accurate, comprehensive, and logically rigorous.

    # Final objective:
    # Provide an optimized report that is concise, structurally clear, and logically coherent. Avoid using clauses or complex sentence structures in the description. Use the provided information to cover all relevant findings. Ensure the report maintains the original report's style and professionalism, while ensuring accuracy and comprehensiveness.

    # Please provide the optimized version of the following report:
    # Global abnormalities: {global_abnormalities}
    # Local abnormalities: {local_abnormalities}
    # Report to optimize: {report}
    # """
    def _build_prompt(self, global_abnormalities: str, local_abnormalities: str, report: str) -> str:

        # ###### PROMPT ORIGIN ######
        # return \
        # f"""
        # You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image model.
        #
        # You must:
        # 1. Incorporate the global abnormalities identified by the image model (e.g., atelectasis, cardiomegaly, pleural effusion, and support devices).
        # 2. Align with the anatomical regions flagged as abnormal (e.g., right lung, left lung, right hemidiaphragm, left hemidiaphragm, and cardiac silhouette).
        # 3. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
        #
        # **Do not include any explanation, notes, or additional comments. Only output the optimized chest X-ray report.**
        #
        # Global abnormalities: {global_abnormalities}
        # Local abnormalities: {local_abnormalities}
        # Report to optimize: {report}
        #
        # Output the optimized report only. Do not include anything else.
        # """
        if self.type == 'PRED_NEWPROMPT_DISEASE_ONLY':
            # ###### disease-only (global abnormalities; no region; positive / negative / blank) ######
            return f"""You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
You must:
1. Incorporate the global abnormalities (disease-level information). Each disease is flagged as positive, negative, or blank by the disease classification model, with a model confidence score. Labels are case-insensitive (positive/Positive, negative/Negative, blank/Blank). Example: ['No Finding', 'blank', 0.958421], ['Cardiomegaly', 'positive', 0.842944], ['Consolidation', 'negative', 0.580098], ['Support Devices', 'positive', 0.544037].
2. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
3. When a positive or negative label conflicts with the original report, that label takes precedence. A blank label does not override the original report.
- Positive: For every disease flagged as positive, explicitly describe its presence using standard radiological terminology for that disease, even if subtle or absent from the original report. (e.g., if Lung Opacity is positive, explicitly describe pulmonary opacity even if subtle.)
- Negative: For every disease flagged as negative, ensure the final report clearly reflects absence or normality using standard negative phrasing (e.g., "no cardiomegaly"; "cardiomediastinal silhouette is not enlarged" / "within normal limits"). If the original report contains explicit positive mentions, equivocal language, or proxy wording that implies the disease, you must revise or remove such wording so the output is consistent with the negative label. Do not preserve proxy terminology that contradicts the negative label.
- Blank: For every disease flagged as blank, the image model did not determine presence or absence. Do not add a new statement that asserts the presence or the absence of that disease. Do not delete, weaken, or reverse an existing mention of that disease in the original report only because its label is blank. Do not write the word "blank" or any confidence score.
4. When multiple diseases are flagged positive, describe each positive disease separately and distinctly; do not merge them into a single vague description. Do not create new descriptions for diseases flagged blank.
Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.
Global abnormalities: {global_abnormalities}  Report to optimize: {report}
Output the optimized report only. Do not include anything else（Also do not use the title like Optimized Chest X-ray Report:, just out put the optimized report only)"""
        elif self.type == 'PRED_NEWPROMPT_DISEASE_ONLY_WBLA':
            # ###### disease-only with blank; positive/negative still override the report ######
            return f"""You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
You must:
1. Incorporate the global abnormalities (disease-level information). Each disease is flagged as positive, negative, or blank by the disease classification model, with a model confidence score (e.g., ['No Finding', 'blank', 0.958421], ['Cardiomegaly', 'positive', 0.842944], ['Consolidation', 'negative', 0.580098], etc.).
2. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
3. When Global abnormalities conflict with the original report, Global abnormalities take precedence. A blank label does not override the original report.
  - Positive: For every disease flagged as positive, explicitly describe its presence using standard radiological terminology for that disease, even if subtle or absent from the original report. (e.g., if Lung Opacity is positive, explicitly describe pulmonary opacity even if subtle.)
  - Negative: For every disease flagged as negative, ensure the final report clearly reflects absence or normality using standard negative phrasing (e.g., "no cardiomegaly"; "cardiomediastinal silhouette is not enlarged" / "within normal limits"). If the original report contains explicit positive mentions, equivocal language, or proxy wording that implies the disease, you must revise or remove such wording so the output is consistent with the negative label. Do not preserve proxy terminology that contradicts the negative label.
4. Blank: For every disease flagged as blank, the image model did not determine presence or absence. Do not add a new statement that asserts the presence or the absence of that disease. Do not delete, weaken, or reverse an existing mention of that disease in the original report only because its label is blank. Do not write the word "blank" or any confidence score.
5. When multiple diseases are flagged positive, describe each positive disease separately and distinctly; do not merge them into a single vague description. Do not create new descriptions for diseases flagged blank.
Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.
Global abnormalities: {global_abnormalities}  Report to optimize: {report}
Output the optimized report only. Do not include anything else（Also do not use the title like Optimized Chest X-ray Report:, just out put the optimized report only)"""
        elif self.type in ('PRED', 'PRED_NEWPROMPT'):
            # ###### PRED / PRED_NEWPROMPT (bidirectional Positive/Negative) ######
            return f"""You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
You must:
1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative identified by the disease classification model with model confidence scores provided (e.g., ['No Finding', 'Negative', 0.9852676391601562], ['Enlarged Cardiomediastinum', 'Negative', 0.940338134765625], ['Pneumothorax', 'Positive', 0.9756], ['Pleural Effusion', 'Positive', 0.824], etc.).
2. Align with the local abnormalities (region-level information), where the anatomical regions are flagged as abnormal or normal identified by the region classification model with confidence scores provided (e.g., ['right lung', 'Abnormal', 0.99025387], ['right upper lung zone', 'Normal', 0.8209005743265152], ['right mid lung zone', 'Abnormal', 0.64220816], ['right lower lung zone', 'Abnormal', 0.92879725], etc.).
3. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
4. When Global abnormalities conflict with the original report, Global abnormalities take precedence.
- Positive: For every disease flagged as Positive, explicitly describe its presence using standard radiological terminology for that disease, even if subtle or absent from the original report.（e.g., if Lung Opacity is Positive, explicitly describe pulmonary opacity even if subtle.)
- Negative: For every disease flagged as Negative, ensure the final report clearly reflects absence or normality using standard negative phrasing (e.g., "no cardiomegaly"; "cardiomediastinal silhouette is not enlarged" / "within normal limits"). If the original report contains explicit positive mentions, equivocal language, or proxy wording that implies the disease, you must revise or remove such wording so the output is consistent with the Negative label. Do not preserve proxy terminology that contradicts the Negative label.
5. When multiple diseases are present, describe each positive disease separately and distinctly; do not merge them into a single vague description.
Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.
Global abnormalities: {global_abnormalities}  Local abnormalities: {local_abnormalities} Report to optimize: {report}
Output the optimized report only. Do not include anything else（Also do not use the title like Optimized Chest X-ray Report:, just out put the optimized report only)"""
        elif self.type == 'LABEL':
            ###### PROMPT WITH LABEL INFO (bidirectional Positive/Negative precedence) ######
            return f"""
You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
You must:
1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative (e.g., ['No Finding', 'Negative'], ['Enlarged Cardiomediastinum', 'Negative'], ['Pneumothorax', 'Positive'], ['Pleural Effusion', 'Positive'], etc.).
2. Align with the local abnormalities (region-level information), where the anatomical regions are flagged as abnormal or normal (e.g., ['right lung', 'Abnormal', ['right upper lung zone', 'Normal'], ['right mid lung zone', 'Abnormal'], ['right lower lung zone', 'Abnormal'], etc.).
3. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
4. When Global abnormalities conflict with the original report, Global abnormalities take precedence.
  - Positive: For every disease flagged as Positive, explicitly describe its presence using standard radiological terminology for that disease, even if subtle or absent from the original report.（e.g., if Lung Opacity is Positive, explicitly describe pulmonary opacity even if subtle.)
  - Negative: For every disease flagged as Negative, ensure the final report clearly reflects absence or normality using standard negative phrasing (e.g., "no cardiomegaly"; "cardiomediastinal silhouette is not enlarged" / "within normal limits"). If the original report contains explicit positive mentions, equivocal language, or proxy wording that implies the disease, you must revise or remove such wording so the output is consistent with the Negative label. Do not preserve proxy terminology that contradicts the Negative label.
  5. When multiple diseases are present, describe each positive disease separately and distinctly; do not merge them into a single vague description.
Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.
Global abnormalities: {global_abnormalities}  Local abnormalities: {local_abnormalities} Report to optimize: {report}
Output the optimized report only. Do not include anything else（Also do not use the title like Optimized Chest X-ray Report:, just out put the optimized report only)
"""
        elif self.type == 'MIX':
            ###### PROMPT WITH MIX INFO ######
            return f"""
            You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.

            You must:
            1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative identified by the disease classification model with model confidence scores provided or not (e.g., ['No Finding', 'Negative', 0.9852676391601562], ['Enlarged Cardiomediastinum', 'Positive', 0.940338134765625], ['Pneumothorax', 'Positive'], ['Pleural Effusion', 'Negative'], etc.).
            2. Align with the local abnormalities (region-level information), where the anatomical regions are flagged as abnormal or normal identified by the region classification model, with confidence scores provided or not (e.g., ['right lung', 'Abnormal', 0.99025387], ['right upper lung zone', 'Normal', 0.8209005743265152], ['right mid lung zone', 'Abnormal'], ['right lower lung zone', 'Abnormal'], etc.).
            3. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.

            **Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.**

            Global abnormalities: {global_abnormalities}
            Local abnormalities: {local_abnormalities}
            Report to optimize: {report}

            Output the optimized report only. Do not include anything else（Also do not use the title like **Optimized Chest X-ray Report:**, just out put the optimized report only)
            """
        elif self.type == 'Glabel_only':
            ###### PROMPT WITH label INFO ######
            return f"""
                You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
    
                You must:
                1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative (e.g., ['No Finding', 'Negative'], ['Enlarged Cardiomediastinum', 'Negative'], ['Pneumothorax', 'Positive'], ['Pleural Effusion', 'Positive'], etc.).
                2. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
    
                **Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.**
    
                Global abnormalities: {global_abnormalities}
                Report to optimize: {report}
    
                Output the optimized report only. Do not include anything else（Also do not use the title like **Optimized Chest X-ray Report:**, just out put the optimized report only)
                """
        elif self.type == 'Gpred_only':
            ###### PROMPT WITH pred INFO ######
            return f"""
                     You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
                    You must:
                    1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative identified by the disease classification model with model confidence scores provided (e.g., ['No Finding', 'Negative', 0.9852676391601562], ['Enlarged Cardiomediastinum', 'Negative', 0.940338134765625], etc.).
                    2. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
            
                    **Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.**
            
                    Global abnormalities: {global_abnormalities}
                    Report to optimize: {report}
            
                    Output the optimized report only. Do not include anything else（Also do not use the title like **Optimized Chest X-ray Report:**, just out put the optimized report only)
            """
        elif self.type == 'LABEL_NEWPROMPT_DISEASE_ONLY':
            return f"""You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models.
You must:
1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative (e.g., ['No Finding', 'Negative'], ['Enlarged Cardiomediastinum', 'Negative'], ['Pneumothorax', 'Positive'], ['Pleural Effusion', 'Positive'], etc.).
2. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
3. When Global abnormalities conflict with the original report, Global abnormalities take precedence.
  - Positive: For every disease flagged as Positive, explicitly describe its presence using standard radiological terminology for that disease, even if subtle or absent from the original report.（e.g., if Lung Opacity is Positive, explicitly describe pulmonary opacity even if subtle.)
  - Negative: For every disease flagged as Negative, ensure the final report clearly reflects absence or normality using standard negative phrasing (e.g., "no cardiomegaly"; "cardiomediastinal silhouette is not enlarged" / "within normal limits"). If the original report contains explicit positive mentions, equivocal language, or proxy wording that implies the disease, you must revise or remove such wording so the output is consistent with the Negative label. Do not preserve proxy terminology that contradicts the Negative label.
4. When multiple diseases are present, describe each positive disease separately and distinctly; do not merge them into a single vague description.
Do not include any explanation, notes, or additional comments. Only output the optimized chest X-ray report as a senior radiologist.
Global abnormalities: {global_abnormalities}  Report to optimize: {report}
Output the optimized report only. Do not include anything else（Also do not use the title like Optimized Chest X-ray Report:, just out put the optimized report only)"""
            # return f"""
            #             You are an expert in optimizing chest X-ray reports. Your task is to improve the chest X-ray report written by a low-level doctor based on the information provided by the image models and experts.
            #
            #             You must:
            #             1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative, and those with confidence scores are identified by the disease classification model, while the remaining ones without confidence scores are idetified by the experts (e.g., ['No Finding', 'Negative', 0.9852676391601562], ['Enlarged Cardiomediastinum', 'Positive', 0.940338134765625], ['Pneumothorax', 'Positive'], ['Pleural Effusion', 'Negative'], etc.).
            #             2. Align with the local abnormalities (region-level information), where the anatomical regions are flagged as abnormal or normal, and those with confidence scores are identified by the region classification model, while the remaining ones without confidence scores are idetified by the experts (e.g., ['right lung', 'Abnormal', 0.99025387], ['right upper lung zone', 'Normal', 0.8209005743265152], ['right mid lung zone', 'Abnormal'], ['right lower lung zone', 'Abnormal'], etc.).
            #             3. Ensure the optimized report maintains the style and professionalism of the original report while being more accurate, concise, and logically clear.
            #
            #             **Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Only output the optimized chest X-ray report as a senior radiologist.**
            #
            #             Global abnormalities: {global_abnormalities}
            #             Local abnormalities: {local_abnormalities}
            #             Report to optimize: {report}
            #
            #             Output the optimized report only. Do not include anything else（Also do not use the title like **Optimized Chest X-ray Report:**, just out put the optimized report only)
            #             """
        else:
            print(f'Invalid report type {self.type}')

    def _parse_output(self, text: str) -> Optional[List[str]]:
        try:
            # 👇 清除Markdown代码块包装
            text = re.sub(r"```(?:python)?", "", text).strip("` \n")
            result = text.strip()
            # print(result)
            # if isinstance(result, list):
            return result
        except Exception as e:
            print("⚠️ Failed to parse model output:", e)
            print("Raw output:", text)
        return None

    async def aextract(self, global_abnormalities: str, report: str, local_abnormalities: str =None, ) -> Optional[List[str]]:
        prompt = self._build_prompt(global_abnormalities, local_abnormalities, report)
        # print(len(prompt))
        # print('\n*********prompt***********\n')
        # print(prompt)
        response = await self.client.chat.completions.create(
            model=self.engine,
            messages=[{"role": "system", "content": "You are a senior radiologist."},
                      {"role": "user", "content": prompt}],
            # max_tokens=self.max_tokens,
            temperature=self.temperature,
            # stream=True
        )

        text = response.choices[0].message.content
        # print('\n*********result from llm***********\n')
        # print(text)
        return self._parse_output(text)

    async def abatch_extract(self, global_abnormalities: List[str], local_abnormalities: List[str],
                             reports: List[str]) -> List[Optional[List[str]]]:
        if local_abnormalities is not None:
            tasks = [self.aextract(global_abnormalities[i], reports[i], local_abnormalities[i]) for i in
                     range(len(reports))]
        else:
            tasks = [self.aextract(global_abnormalities[i], reports[i]) for i in
                     range(len(reports))]
        return await asyncio.gather(*tasks)

    def _build_prompt_multi_baseline(
        self,
        global_abnormalities: str,
        local_abnormalities: str,
        reports_by_method: dict,
        methods: List[str],
        result_prefix: str = "llm_report_conflict_knn_",
    ) -> str:
        report_blocks = []
        json_keys = []
        for method in methods:
            report_col = reports_by_method.get(method, "")
            report_blocks.append(
                f"[{method} baseline report]\n{report_col}\n"
            )
            json_keys.append(f'"{result_prefix}{method}"')

        reports_text = "\n".join(report_blocks)
        json_schema = ",\n".join(f'  {k}: "<optimized report for this method only>"' for k in json_keys)

        if self.type == "PRED":
            return f"""
You are an expert in optimizing chest X-ray reports. Your task is to improve chest X-ray reports written by low-level doctors based on the information provided by the image models.

You are given ONE shared set of image-model outputs and FIVE independent baseline reports from different report-generation methods. You must optimize EACH baseline report independently.

Shared inputs (same for all methods):
1. Incorporate the global abnormalities (disease-level information), where the disease are flagged as positive or negative identified by the disease classification model with model confidence scores provided (e.g., ['No Finding', 'Negative', 0.9852676391601562], ['Enlarged Cardiomediastinum', 'Negative', 0.940338134765625], ['Pneumothorax', 'Positive', 0.9756], ['Pleural Effusion', 'Positive', 0.824], etc.).
2. Align with the local abnormalities (region-level information), where the anatomical regions are flagged as abnormal or normal identified by the region classification model with confidence scores provided (e.g., ['right lung', 'Abnormal', 0.99025387], ['right upper lung zone', 'Normal', 0.8209005743265152], ['right mid lung zone', 'Abnormal', 0.64220816], ['right lower lung zone', 'Abnormal', 0.92879725], etc.).

Global abnormalities: {global_abnormalities}
Local abnormalities: {local_abnormalities}

Baseline report group (five methods, listed separately; optimize each independently):
{reports_text}

For EACH method independently, you must:
3. Ensure the optimized report maintains the style and professionalism of THAT method's own original report while being more accurate, concise, and logically clear.
4. When Global abnormalities conflict with the original report, Global abnormalities take precedence.
   - Positive: For every disease flagged as Positive, explicitly describe its presence using standard radiological terminology for that disease, even if subtle or absent from the original report. (e.g., if Lung Opacity is Positive, explicitly describe pulmonary opacity even if subtle.)
   - Negative: For every disease flagged as Negative, ensure the final report clearly reflects absence or normality using standard negative phrasing (e.g., "no cardiomegaly"; "cardiomediastinal silhouette is not enlarged" / "within normal limits"). If the original report contains explicit positive mentions, equivocal language, or proxy wording that implies the disease, you must revise or remove such wording so the output is consistent with the Negative label. Do not preserve proxy terminology that contradicts the Negative label.
5. When multiple diseases are present, describe each positive disease separately and distinctly; do not merge them into a single vague description.

Critical independence rules:
- Treat each method as a fully independent optimization task.
- Do NOT copy wording/findings from other methods' baseline reports into another method's output.
- Do NOT merge, average, or compromise across methods.
- Each output must reflect only its own baseline report plus the shared global/local abnormalities.

Output format:
Return ONLY valid JSON (no markdown, no explanation) with exactly these keys:
{{
{json_schema}
}}

Do not include any explanation, notes, or additional comments. Do not include any description about confidence score in the output text. Do not use titles like "Optimized Chest X-ray Report". Only output the optimized reports in the JSON object above.
"""
        raise ValueError(f"Multi-baseline prompt not implemented for type={self.type}")

    def _parse_multi_output(
        self,
        text: str,
        methods: List[str],
        result_prefix: str = "llm_report_conflict_knn_",
    ) -> dict:
        if text is None:
            return {f"{result_prefix}{m}": None for m in methods}
        cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                print("⚠️ Failed to parse multi-baseline JSON:", cleaned[:500])
                return {f"{result_prefix}{m}": None for m in methods}
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print("⚠️ Failed to parse multi-baseline JSON:", e)
                print("Raw output:", cleaned[:500])
                return {f"{result_prefix}{m}": None for m in methods}

        out = {}
        for method in methods:
            key = f"{result_prefix}{method}"
            val = obj.get(key)
            if val is None:
                alt = obj.get(method)
                val = alt
            out[key] = self._parse_output(str(val)) if val is not None else None
        return out

    async def aextract_multi_baseline(
        self,
        global_abnormalities: str,
        local_abnormalities: str,
        reports_by_method: dict,
        methods: List[str],
        result_prefix: str = "llm_report_conflict_knn_",
    ) -> dict:
        prompt = self._build_prompt_multi_baseline(
            global_abnormalities,
            local_abnormalities,
            reports_by_method,
            methods,
            result_prefix=result_prefix,
        )
        response = await self.client.chat.completions.create(
            model=self.engine,
            messages=[
                {"role": "system", "content": "You are a senior radiologist."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        return self._parse_multi_output(text, methods, result_prefix=result_prefix)

    async def abatch_extract_multi_baseline(
        self,
        global_abnormalities: List[str],
        local_abnormalities: List[str],
        reports_by_method_list: List[dict],
        methods: List[str],
        result_prefix: str = "llm_report_conflict_knn_",
    ) -> List[dict]:
        tasks = [
            self.aextract_multi_baseline(
                global_abnormalities[i],
                local_abnormalities[i],
                reports_by_method_list[i],
                methods,
                result_prefix=result_prefix,
            )
            for i in range(len(reports_by_method_list))
        ]
        return await asyncio.gather(*tasks)


DEFAULT_FIVE_METHODS = ["CMCRL", "DCL", "EKAGEN", "TOKENMIXER", "promptMRG"]


async def process_csv_async_multi_baseline(
    api_key: str,
    base_url: str,
    model: str,
    input_csv: str,
    output_csv: str,
    methods: Optional[List[str]] = None,
    report_prefix: str = "predict_report_",
    result_prefix: str = "llm_report_conflict_knn_",
    global_column: str = "pred_disease_info_chexber_uone_15w_knn_new",
    local_column: str = "pred_region_info_15w_knn_new_region_conflicts_knn",
    batch_size: int = 20,
    sample_len: Optional[int] = None,
    type: str = "PRED",
    proxy: Optional[str] = None,
    path_column: str = "id",
    filter_column: Optional[str] = "need_regenerate_report",
    filter_value=True,
):
    import httpx

    methods = list(methods or DEFAULT_FIVE_METHODS)
    result_columns = [f"{result_prefix}{m}" for m in methods]
    report_columns = [f"{report_prefix}{m}" for m in methods]

    extractor = Refined_Report(
        api_key=api_key, base_url=base_url, engine=model, max_tokens=512, type=type
    )
    if proxy:
        timeout = httpx.Timeout(120.0, connect=30.0)
        extractor.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(proxy=proxy, timeout=timeout),
        )

    if input_csv.endswith(".xlsx"):
        df_all = pd.read_excel(input_csv, engine="openpyxl")
    elif input_csv.endswith(".csv"):
        df_all = pd.read_csv(input_csv, low_memory=False)
    else:
        raise ValueError(f"Unsupported input file: {input_csv}")

    if path_column not in df_all.columns:
        if "id" in df_all.columns:
            path_column = "id"
        elif "Path" in df_all.columns:
            path_column = "Path"
        else:
            raise KeyError(f"Missing path column in {input_csv}")

    if filter_column is not None:
        mask = df_all[filter_column].astype(str).str.lower().isin(
            {"true", "1", "需要", "yes"}
        ) | (df_all[filter_column] == True)
        df_all = df_all.loc[mask].copy()
        print(f"Filter {filter_column}=True/需要: {len(df_all)} rows")

    for col in report_columns + [global_column, local_column]:
        if col not in df_all.columns:
            raise KeyError(f"Missing column: {col}")

    output_dir = os.path.dirname(output_csv) or "."
    output_subpath = Path(output_csv).stem
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv, low_memory=False)
        for rc in result_columns:
            if rc not in df.columns:
                df[rc] = pd.NA
        missing_mask = df[result_columns[0]].isna() | (
            df[result_columns[0]].astype(str).str.strip().isin(("", "nan", "None"))
        )
        df_new = df.loc[missing_mask].copy()
        print(f"Resume {output_csv}: missing {len(df_new)} / {len(df)} rows")
    else:
        df = df_all.copy()
        if sample_len is not None:
            df = df.head(sample_len).copy()
        for rc in result_columns:
            df[rc] = pd.NA
        df.to_csv(output_csv, index=False)
        df_new = df.copy()
        print(f"New run: {len(df)} rows (sample_len={sample_len})")

    if global_column is not None:
        df_new[global_column] = df_new[global_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    if local_column is not None:
        df_new[local_column] = df_new[local_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    for i in range(0, len(df_new), batch_size):
        batch_df = df_new.iloc[i : i + batch_size]
        paths = batch_df[path_column].tolist()
        batch_globals = batch_df[global_column].tolist()
        batch_locals = batch_df[local_column].tolist()
        reports_by_method_list = []
        for _, row in batch_df.iterrows():
            reports_by_method_list.append(
                {m: row[f"{report_prefix}{m}"] for m in methods}
            )

        print(f"📦 Processing rows {i} to {i + len(batch_df)}...")
        start_time = time.time()
        try:
            batch_results = await extractor.abatch_extract_multi_baseline(
                batch_globals,
                batch_locals,
                reports_by_method_list,
                methods,
                result_prefix=result_prefix,
            )
        except Exception as e:
            print(f"❌ Batch {i} failed: {e}")
            batch_results = [{rc: None for rc in result_columns} for _ in batch_df.index]
            with open(
                os.path.join(output_dir, f"{output_subpath}_partial_{i}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(list(zip(paths, batch_results)), f, indent=2, ensure_ascii=False)

        batch_idx = batch_df.index
        for rc in result_columns:
            df.loc[batch_idx, rc] = [br.get(rc) for br in batch_results]
        # atomic save to avoid corrupt CSV on interrupt/concurrent read
        tmp_out = output_csv + ".tmp"
        df.to_csv(tmp_out, index=False)
        os.replace(tmp_out, output_csv)
        ok = sum(
            all(
                br.get(rc) is not None and str(br.get(rc)).strip() not in ("", "None")
                for rc in result_columns
            )
            for br in batch_results
        )
        print(
            f"Inference time for batch: {time.time() - start_time:.4f}s, "
            f"full_ok={ok}/{len(batch_results)}"
        )

    print(f"✅ Multi-baseline extraction complete. Results saved to {output_csv}")
    return df


def _load_table(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        kwargs = {"engine": "openpyxl"}
        if sheet_name is not None:
            kwargs["sheet_name"] = sheet_name
        return pd.read_excel(path, **kwargs)
    if path.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table path: {path}")


def _save_table(df: pd.DataFrame, path: str) -> None:
    """Atomic save to csv or xlsx based on extension."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    if path.endswith(".xlsx"):
        tmp = path + ".tmp.xlsx"
        df.to_excel(tmp, index=False, engine="openpyxl")
        os.replace(tmp, path)
    elif path.endswith(".csv"):
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    else:
        raise ValueError(f"Unsupported output path: {path}")


async def process_csv_async(
        api_key: str,
        base_url: str,
        model:str,
        # temperature: float,
        input_csv: str,
        output_csv: str,
        report_column: str = "res",
        global_column: str = "chexpert_labels",
        local_column: str = "structured_findings",
        result_column: str = "gpt",
        batch_size: int = 10,
        sample_len = None,
        type = 'PRED',
        proxy: Optional[str] = None,
        sheet_name: Optional[str] = None,
        path_column: str = "Path",
):
    import httpx

    extractor = Refined_Report(api_key=api_key, base_url=base_url, engine=model, max_tokens=512, type=type)
    if proxy:
        timeout = httpx.Timeout(120.0, connect=30.0)
        extractor.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(proxy=proxy, timeout=timeout),
        )

    try:
        df = _load_table(input_csv, sheet_name=sheet_name)
    except ValueError:
        print('error input file name {}'.format(input_csv))
        return None

    if path_column not in df.columns:
        if path_column == "Path" and "id" in df.columns:
            df["Path"] = df["id"].astype(str)
            path_column = "Path"
        elif "id" in df.columns:
            path_column = "id"
        elif "ID" in df.columns:
            path_column = "ID"
        else:
            raise KeyError(f"Missing path column in {input_csv}")

    # 若output文件存在（即之前生成过），则读取df中结果不为na的元组

    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)
    output_subpath = Path(output_csv).stem
    if os.path.exists(output_csv):
        df = _load_table(output_csv)
        if result_column not in df.columns:
            df[result_column] = pd.NA
        missing_rows = df[df[result_column].isna() | (
            df[result_column].astype(str).str.strip().isin(("", "nan", "None"))
        )]
        df_new = missing_rows.copy()
        print(f'FIle {output_csv} 已存在，其中未生成元组有{len(missing_rows)}个')
    else:
        if sample_len is not None:
            df = df.head(sample_len).copy()
        df_new = df.copy()
        if result_column not in df.columns:
            df[result_column] = pd.NA
        print(f'input {input_csv} 包含 {len(df)}个元组 (sample_len={sample_len})')
    # df = df[df['available'] == 'yes'] # available 指元组至少在一个标签上为1
    # df = df[df['gpt_output_label'] == 'BLANK']
    # df[report_column] = df[report_column].apply(ast.literal_eval)
    if global_column is not None:
        df_new[global_column] = df_new[global_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    if local_column is not None:
        df_new[local_column] = df_new[local_column].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    all_results = []
    all_results_temp = []
    # length = sample_len if sample_len < len(df) else len(df)
    # sample_len already applied on new runs; keep slice for resume+sample smoke
    if sample_len is not None and len(df_new) > sample_len and not os.path.exists(output_csv):
        df_new = df_new[:sample_len]

    for i in range(0, len(df_new), batch_size):

        paths = df_new[path_column].iloc[i:i + batch_size].tolist()
        # for i in range(0, 1, batch_size):
        batch_reports = df_new[report_column].iloc[i:i + batch_size].tolist()
        batch_globals = df_new[global_column].iloc[i:i + batch_size].tolist()
        if local_column is not None:
            batch_locals = df_new[local_column].iloc[i:i + batch_size].tolist()
        else:
            batch_locals = None

        print(f"📦 Processing reports {i} to {i + len(batch_reports)}...")

        start_time = time.time()
        try:
            # 批量提取优化报告
            batch_results = await extractor.abatch_extract(batch_globals, batch_locals, batch_reports)
        except Exception as e:
            print(f"❌ Batch {i} failed: {e}")
            batch_results = [None] * len(batch_reports)
            with open(output_dir + f"/{output_subpath}_partial_results_utill_{i}.json", "w") as f:
                json.dump(all_results_temp, f, indent=2)

        all_results.extend(batch_results)
        all_results_temp.extend(zip(paths, batch_results))

        elapsed_time = time.time() - start_time
        print(f"Inference time for batch: {elapsed_time:.4f} seconds", flush=True)

        # incremental checkpoint (every 10 batches, or last batch)
        done_n = len(all_results)
        batch_i = i // batch_size
        is_last = done_n >= len(df_new)
        if (batch_i % 10 == 0) or is_last or done_n == len(df_new):
            assign_idx = df_new.index[:done_n]
            if result_column not in df.columns:
                df[result_column] = pd.NA
            df.loc[assign_idx, result_column] = all_results
            _save_table(df, output_csv)
            filled = int((~df[result_column].isna() & ~df[result_column].astype(str).str.strip().isin(("", "nan", "None"))).sum())
            print(
                f"💾 checkpoint saved: batch_done={done_n}/{len(df_new)} "
                f"filled≈{filled}/{len(df)} -> {output_csv}",
                flush=True,
            )
        else:
            print(
                f"… progress batch_done={done_n}/{len(df_new)} "
                f"({100.0 * done_n / max(1, len(df_new)):.2f}%)",
                flush=True,
            )
        # lightweight progress sidecar
        try:
            prog_path = os.path.join(output_dir, f"{output_subpath}_progress.txt")
            with open(prog_path, "w") as pf:
                pf.write(
                    f"done={done_n}\ntotal={len(df_new)}\n"
                    f"pct={100.0 * done_n / max(1, len(df_new)):.4f}\n"
                )
        except Exception:
            pass
        # with open(output_dir + f"/partial_results_utill_{i}.json", "w") as f:
        #     json.dump(all_results_temp, f, indent=2)
    # final assignment + save
    if result_column not in df.columns:
        df[result_column] = pd.NA
    if len(all_results) != len(df_new):
        print(
            f"⚠️ result length mismatch: got {len(all_results)} expected {len(df_new)}",
            flush=True,
        )
    df.loc[df_new.index[: len(all_results)], result_column] = all_results
    _save_table(df, output_csv)
    print(f"✅ Extraction complete. Results saved to {output_csv}", flush=True)
    return df

def df_merge():
    # df_test = pd.read_excel('./data/<path>', engine='openpyxl')
    # df_chexbert_15w = pd.read_excel('./data/<path>', engine = 'openpyxl')
    # df_new = pd.merge(df_test, df_chexbert_15w, on='mimic_image_file_path', how = 'left')
    # df_new.to_excel('./data/<path>', index=False)

    # df_info = pd.read_excel('./data/<path>', engine='openpyxl')
    # df_forLLM = pd.read_csv('./data/<path>')
    # df_new = pd.merge(df_forLLM, df_info[['dicom_id', 'pred_region_info', '10w_uone_chexbert_pred_info']], left_on = 'id', right_on = 'dicom_id')
    # df_new.to_csv('./data/<path>')

    df_disease = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine="openpyxl")
    df_region = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine="openpyxl")
    df = pd.read_csv(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))))
    df_region['id'] = df_region['mimic_image_file_path'].apply(lambda x: (x.split('/')[-1]).split('.')[0])
    df_disease['id'] = df_disease['mimic_image_file_path'].apply(lambda x: (x.split('/')[-1]).split('.')[0])
    df_new = pd.merge(df, df_region[["id", "region_info"]], on="id", how="left")
    df_new = pd.merge(df_new, df_disease[["id", "pred_info"]], on="id", how="left")
    df_new['gt_report_clean'] = df_new['gt_report'].apply(clean_report_mimic_cxr)
    df_new['r2gen_report_clean'] = df_new['r2gen_report'].apply(clean_report_mimic_cxr)
    df_new.to_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine="openpyxl", index=False)
    return None

def info_merge():
    Label_chexBert = ["No Finding_y",
                        "Enlarged Cardiomediastinum_y",
                        "Cardiomegaly_y",
                        "Lung Opacity_y",
                        "Lung Lesion_y",
                        "Edema_y",
                        "Consolidation_y",
                        "Pneumonia_y",
                        "Atelectasis_y",
                        "Pneumothorax_y",
                        "Pleural Effusion_y",
                        "Pleural Other_y",
                        "Fracture_y",
                        "Support Devices_y"]
    ANATOMICAL_REGIONS = {
        "right lung": 0,
        "right upper lung zone": 1,
        "right mid lung zone": 2,
        "right lower lung zone": 3,
        "right hilar structures": 4,
        "right apical zone": 5,
        "right costophrenic angle": 6,
        "right hemidiaphragm": 7,
        "left lung": 8,
        "left upper lung zone": 9,
        "left mid lung zone": 10,
        "left lower lung zone": 11,
        "left hilar structures": 12,
        "left apical zone": 13,
        "left costophrenic angle": 14,
        "left hemidiaphragm": 15,
        "trachea": 16,
        "spine": 17,
        "right clavicle": 18,
        "left clavicle": 19,
        "aortic arch": 20,
        "mediastinum": 21,
        "upper mediastinum": 22,
        "svc": 23,
        "cardiac silhouette": 24,
        "cavoatrial junction": 25,
        "right atrium": 26,
        "carina": 27,
        "abdomen": 28
    }
    # df = pd.read_csv('./data/<path>')
    df = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine='openpyxl')
    df.fillna("NA", inplace=True)
    df['bbox_is_abnormal'] = df['bbox_is_abnormal'].apply(ast.literal_eval)
    label_region = []
    label_disease = []
    REGIONS = list(ANATOMICAL_REGIONS.keys())
    for index, row in df.iterrows():
        bbox_is_abnormal = row['bbox_is_abnormal']
        region_label = []
        disease_label = []
        bool2normal = {True: 'Abnormal', False: 'Normal'}
        for i, ba in enumerate(bbox_is_abnormal):
            region_label.append([REGIONS[i], bool2normal[ba]])
        for lb in Label_chexBert:
            lb_data = row[lb]
            if lb_data == 1.0 or lb_data == -1.0:
                disease_label.append([lb.split('_')[0],'Positive'])
            else:
                disease_label.append([lb.split('_')[0],'Negative'])
        label_region.append(region_label)
        label_disease.append(disease_label)
    df.insert(0, 'label_region', label_region)
    df.insert(1, 'label_disease', label_disease)
    df.to_csv(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))))

def translate_en_to_zh(text):
    translated = GoogleTranslator(source='en', target='zh-CN').translate(text)
    print("翻译：", translated)
    return translated

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompted LLM report refinement")
    parser.add_argument("--input_csv", required=True, help="Input table (csv/xlsx)")
    parser.add_argument("--output_csv", required=True, help="Output csv path")
    parser.add_argument("--report_column", default="r2gen_report")
    parser.add_argument("--global_column", default="pred_disease_info_chexber_uone_15w")
    parser.add_argument("--local_column", default="pred_region_info_15w")
    parser.add_argument("--result_column", default="llm_report")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4.1-mini"))
    parser.add_argument(
        "--base_url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please export OPENAI_API_KEY before running.")

    asyncio.run(
        process_csv_async(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            batch_size=args.batch_size,
            report_column=args.report_column,
            global_column=args.global_column,
            local_column=args.local_column,
            result_column=args.result_column,
        )
    )

