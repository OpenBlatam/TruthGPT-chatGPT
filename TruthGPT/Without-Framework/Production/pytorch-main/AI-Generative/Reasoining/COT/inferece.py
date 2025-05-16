import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import re
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import uuid
import copy
import json
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True
        return False
    
class StopOnPeriod(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated_text.endswith('.'):
            return True
        return False
    
class llama_vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    # This function is used to split Llama-3.2-90B
    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size

        num_layers = 100
        # GPU0: -5, GPU-1: -7
        total_cost = num_layers + 5 + 7

        # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
        num_layers_per_gpu = total_cost // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        num_layers_per_gpu[0] -= 5
        num_layers_per_gpu[-1] -= 7

        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        device_map['vision_model'] = rank
        device_map['language_model.model.embed_tokens'] = rank
        device_map['language_model.model.rotary_emb'] = rank
        device_map['language_model.model.norm'] = rank + world_size * (num_gpus - 1)
        device_map['language_model.lm_head'] = rank + world_size * (num_gpus - 1)
        device_map['multi_modal_projector'] = rank + world_size * (num_gpus - 1)
        return device_map

    def __init__(self, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct', **kwargs):
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
        except Exception as e:
            logging.critical('Please install transformers>=4.45.0 before using llama_vision.')
            raise e

        if '90b' in model_path.lower():
            device_map = self.split_model()
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            ).eval()
        else:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
            ).cuda().eval()

        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_path)
        if 'Instruct' in model_path:
            kwargs_default = dict(do_sample=True, temperature=0.6, top_p=0.9)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=5)
        kwargs_default = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)
        kwargs.update(kwargs_default)
        print(f'Following kwargs received: {kwargs}, will use as generation config. ')
        self.kwargs = kwargs
        self.model_name = model_path

    def use_custom_prompt(self, dataset):
        if dataset is None:
            return False
        if listinstr(['AI2D', 'MMMU', 'MathVista', 'ChartQA', 'DocVQA'], dataset):
            # For Certain dataset we use custom prompt
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        if listinstr(['AI2D'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            for key, item in options.items():
                question += f'\n{key}. {item}'
            if '11B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'Look at the scientific diagram carefully and answer the following question: {question}\n'
                    f'Respond only with the correct option digit.'
                )
        elif listinstr(['MMMU'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            options = '\n'.join([f'{key}. {item}' for key, item in options.items()])
            prompt = (
                f'Look at the image carefully and solve the following question step-by-step. '
                f'Question: {question} Options: {options} Indicate the correct answer at the end.'
            )
            for i in range(len(tgt_path)):
                prompt = prompt.replace(f'<image {i+1}>', '')
        elif listinstr(['MathVista'], dataset):
            self.kwargs['max_new_tokens'] = 2048
            prompt = f'{question}'
        elif listinstr(['ChartQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            if '11B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'You have to think through your answer and provide a step-by-step solution. '
                    f'Once you have the solution, write the final answer in at most a few words at the end '
                    f"with the phrase \"FINAL ANSWER:\". "
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
            elif '90B' in self.model_name:
                prompt = (
                    f'You are provided a chart image and will be asked a question. '
                    f'Follow these steps carefully:\n '
                    f'Step 1: Analyze the question to understand what specific data or information is being asked for. '
                    f'Focus on whether the question is asking for a specific number or category '
                    f'from the chart image.\n '
                    f'Step 2: Identify any numbers, categories, or groups mentioned in the question '
                    f'and take note of them. Focus on detecting and matching them directly to the image. \n'
                    f'Step 3: Study the image carefully and find the relevant data corresponding to the categories '
                    f'or numbers mentioned. Avoid unnecessary assumptions or calculations; '
                    f'simply read the correct data from the image.\n '
                    f'Step 4: Develop a clear plan to solve the question by locating the right data. '
                    f'Focus only on the specific category or group that matches the question. \n'
                    f'Step 5: Use step-by-step reasoning to ensure you are referencing the correct numbers '
                    f'or data points from the image, avoiding unnecessary extra steps or interpretations.\n '
                    f"Step 6: Provide the final answer, starting with \"FINAL ANSWER:\" "
                    f'and using as few words as possible, '
                    f'simply stating the number or data point requested. \n\n '
                    f"The question is: {question}<cot_start>Let's think step by step."
                )
        elif listinstr(['DocVQA'], dataset):
            self.kwargs['max_new_tokens'] = 512
            prompt = (
                f'Read the text in the image carefully and answer the question '
                f'with the text as seen exactly in the image. '
                f'For yes/no questions, just respond Yes or No. '
                f'If the answer is numeric, just respond with the number and nothing else. '
                f'If the answer has multiple words, just respond with the words and absolutely nothing else. '
                f'Never respond in a sentence or a phrase.\n Question: {question}'
            )
        else:
            raise NotImplementedError(f'Dataset {dataset}) not supported.')

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
    
    def judge(self, image, prompt, outputs, type="summary"):
        input_outputs = []
        
        hint = None
        if type == "all":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better answers the question.'
            recall_prompt = ""
            for output in outputs:
                input_outputs.append(output)
        elif type == "sentence":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide is a better next sentence for the answer to the question.'
            recall_prompt = ""
            for output in outputs:
                sentences = output.split(".")
                if len(sentences) > 2:
                    hint = ' '.join(sentences[:-2])
                input_outputs.append(sentences[-2])
        elif type == "summary":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better provides a summary of what it should do to solve the question. The summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
            recall_prompt = f'Please note that a better summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
            for output in outputs:
                input_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                if input_match:
                    input_outputs.append(input_match.group(1))
        elif type == "caption":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better summarizes the information in the image related to the question, and has fewer errors. It is essential that the captions are as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
            recall_prompt = f'Please note that a better caption should be as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
            for output in outputs:
                input_match = re.search(r'<CAPTION>(.*?)</CAPTION>', output, re.DOTALL)
                if input_match:
                    hint_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                    if hint_match:
                        input_outputs.append(input_match.group(1))
        elif type == "reasoning":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better explains the reasoning process to solve the question, and has fewer errors. Begin by thoroughly reviewing the question, followed by an in-depth examination of each answer individually, noting any differences. Subsequently, analyze these differences to determine which response demonstrates stronger reasoning and provide a clear conclusion.'
            recall_prompt = f'Begin by thoroughly reviewing the question, followed by an in-depth examination of each answer individually, noting any differences. Subsequently, analyze these differences to determine which response demonstrates stronger reasoning and provide a clear conclusion.'
            for output in outputs:
                input_match = re.search(r'<REASONING>(.*?)</REASONING>', output, re.DOTALL)
                if input_match:
                    hint_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                    if hint_match:
                        hint_caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', output, re.DOTALL)
                        if hint_caption_match:
                            hint = hint_caption_match.group(1)
                            input_outputs.append(input_match.group(1))
        elif type == "conclusion":
            judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide offers a more effective conclusion to the question. The conclusion should align with the reasoning presented in the hint. The conclusion should never refuse to answer the question.'
            recall_prompt = f'Please note that a better conclusion should align with the reasoning presented in the hint. The conclusion should never refuse to answer the question.'
            for output in outputs:
                input_match = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', output, re.DOTALL)
                if input_match:
                    hint_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                    if hint_match:
                        hint_caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', output, re.DOTALL)
                        if hint_caption_match:
                            hint_reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', output, re.DOTALL)
                            if hint_reasoning_match:
                                hint = hint_caption_match.group(1) + hint_reasoning_match.group(1)
                                input_outputs.append(input_match.group(1))

        if type == "reasoning":
            reasoning_prompt = f"""Now you act as a judge, helping me determine whether the reasoning process in the given text is correct and accurate based on the given information.
            You should assume that the given information about the image is correct.
            You should only consider the reasoning process itself, not the correctness of the background information.  
            If the reasoning process invovles any calculations, you should verify the accuracy of the calculations.
            You should output 'correct' if you don't find any errors in the reasoning process, and 'incorrect' if you find any errors."""
            
            reasoning_prompt_1 = reasoning_prompt + f'\n\nGiven Information: {hint}' + f'\n\nReasoning Process: {input_outputs[0]}'
            reasoning_message_1 = [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': reasoning_prompt_1}
                ]}
            ]
            reasoning_input_text_1 = self.processor.apply_chat_template(reasoning_message_1, add_generation_prompt=True)
            reasoning_inputs_1 = self.processor(None, reasoning_input_text_1, return_tensors='pt').to(self.device)
            reasoning_output_1 = self.model.generate(**reasoning_inputs_1, **self.kwargs)
            reasoning_output_text_1 = self.processor.decode(reasoning_output_1[0][reasoning_inputs_1['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
            if "incorrect" in reasoning_output_text_1:
                #logging
                with open('log.jsonl', 'a') as f:
                    json_obj = {
                        "prompt": prompt,
                        "outputs": outputs,
                        "judge_output": reasoning_output_text_1
                    }
                    f.write(json.dumps(json_obj) + '\n')
                return 1
            
            reasoning_prompt_2 = reasoning_prompt + f'\n\nGiven Information: {hint}' + f'\n\nReasoning Process: {input_outputs[1]}'
            reasoning_message_2 = [
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': reasoning_prompt_2}
                ]}
            ]
            reasoning_input_text_2 = self.processor.apply_chat_template(reasoning_message_2, add_generation_prompt=True)
            reasoning_inputs_2 = self.processor(None, reasoning_input_text_2, return_tensors='pt').to(self.device)
            reasoning_output_2 = self.model.generate(**reasoning_inputs_2, **self.kwargs)
            reasoning_output_text_2 = self.processor.decode(reasoning_output_2[0][reasoning_inputs_2['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
            if "incorrect" in reasoning_output_text_2:
                #logging
                with open('log.jsonl', 'a') as f:
                    json_obj = {
                        "prompt": prompt,
                        "outputs": outputs,
                        "judge_output": reasoning_output_text_2
                    }
                    f.write(json.dumps(json_obj) + '\n')
                return 0
                
        judge_prompt += f'\n\nQuestion: {prompt}'
        if hint:
            judge_prompt += f'\n\nHint about the Question: {hint}'
        for i, output in enumerate(input_outputs):
            judge_prompt += f'\nRepsonse {i+1}: {output}'
        judge_prompt += f'\n\n{recall_prompt}'
        judge_prompt += f' Please strictly follow the following format requirements when outputting, and don’t have any other unnecessary words.'
        judge_prompt += f'\n\nOutput format: "Since [reason], I choose response [1/2]."'
        
        judge_message = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': judge_prompt}
            ]}
        ]
        judge_input_text = self.processor.apply_chat_template(judge_message, add_generation_prompt=True)
        judge_inputs = self.processor(image, judge_input_text, return_tensors='pt').to(self.device)
        judge_output = self.model.generate(**judge_inputs, **self.kwargs)
        judge_output_text = self.processor.decode(judge_output[0][judge_inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
        
        # log to log.jsonl (json format){"prompt": prompt, "outputs": outputs, "judge_output": judge_output_text}
        with open('log.jsonl', 'a') as f:
            json_obj = {
                "prompt": prompt,
                "outputs": outputs,
                "judge_output": judge_output_text
            }
            f.write(json.dumps(json_obj) + '\n')
        
        if "I choose response 1" in judge_output_text:
            return 0
        else:
            return 1
        
    def generate_inner_best_of_N(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        if not self.use_custom_prompt(dataset):
            if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N':
                self.kwargs['max_new_tokens'] = 2048
            else:
                self.kwargs['max_new_tokens'] = 2048
        
        initial_length = len(inputs['input_ids'][0])
        input_ids = copy.deepcopy(inputs['input_ids'])

        stop_criteria = StoppingCriteriaList([StopOnStrings(['</CONCLUSION>'], self.processor.tokenizer)])   
        candidates = []
        for _ in range(10):  
            generation_kwargs = self.kwargs.copy()
            generation_kwargs.update({
                'stopping_criteria': stop_criteria
            })
            
            inputs = self.processor(image, input_ids, return_tensors='pt').to(self.device)
            output = self.model.generate(**inputs, **generation_kwargs)
            
            new_generated_ids = output[0]
            
            generated_text = self.processor.tokenizer.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
            
            candidates.append({
                'input_ids': new_generated_ids.unsqueeze(0),
                'generated_text': generated_text,
            })
        
        while(len(candidates) > 1):
            # randomly select two candidates
            candidate1 = candidates.pop(np.random.randint(len(candidates)))
            candidate2 = candidates.pop(np.random.randint(len(candidates)))
            outputs = [candidate1['generated_text'], candidate2['generated_text']]
            best_index = self.judge(image, prompt, outputs, type="all")
            if best_index == 0:
                candidates.append(candidate1)
            else:
                candidates.append(candidate2)
        
        input_ids = candidates[0]['input_ids']

        final_output = self.processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True)
        return final_output
    
    def generate_inner_sentence_beam(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        if not self.use_custom_prompt(dataset):
            if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N':
                self.kwargs['max_new_tokens'] = 2048
            else:
                self.kwargs['max_new_tokens'] = 2048

        initial_length = len(inputs['input_ids'][0])
        input_ids = copy.deepcopy(inputs['input_ids'])

        while "</CONCLUSION>" not in self.processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True):
            stop_criteria = StoppingCriteriaList([StopOnPeriod(self.processor.tokenizer), StopOnStrings(["</CONCLUSION>"], self.processor.tokenizer)])
            
            candidates = []
            for _ in range(5):  
                generation_kwargs = self.kwargs.copy()
                generation_kwargs.update({
                    'stopping_criteria': stop_criteria
                })
                
                inputs = self.processor(image, input_ids, return_tensors='pt').to(self.device)
                output = self.model.generate(**inputs, **generation_kwargs)
                
                new_generated_ids = output[0]
                
                generated_text = self.processor.tokenizer.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
                
                candidates.append({
                    'input_ids': new_generated_ids.unsqueeze(0),
                    'generated_text': generated_text,
                })
            
            while(len(candidates) > 1):
                # randomly select two candidates
                candidate1 = candidates.pop(np.random.randint(len(candidates)))
                candidate2 = candidates.pop(np.random.randint(len(candidates)))
                outputs = [candidate1['generated_text'], candidate2['generated_text']]
                best_index = self.judge(image, prompt, outputs, type="sentence")
                if best_index == 0:
                    candidates.append(candidate1)
                else:
                    candidates.append(candidate2)
            
            input_ids = candidates[0]['input_ids']

        final_output = self.processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True)
        return final_output
    
    def generate_inner_stage_beam(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors='pt').to(self.device)
        if not self.use_custom_prompt(dataset):
            if DATASET_TYPE(dataset) == 'MCQ' or DATASET_TYPE(dataset) == 'Y/N':
                self.kwargs['max_new_tokens'] = 2048
            else:
                self.kwargs['max_new_tokens'] = 2048
        
        stages = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
        end_markers = ['</SUMMARY>', '</CAPTION>', '</REASONING>', '</CONCLUSION>']

        initial_length = len(inputs['input_ids'][0])
        input_ids = copy.deepcopy(inputs['input_ids'])

        for stage, end_marker in zip(stages, end_markers):
            stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], self.processor.tokenizer)])
            
            candidates = []
            for _ in range(10):  
                generation_kwargs = self.kwargs.copy()
                generation_kwargs.update({
                    'stopping_criteria': stop_criteria
                })
                
                inputs = self.processor(image, input_ids, return_tensors='pt').to(self.device)
                output = self.model.generate(**inputs, **generation_kwargs)
                
                new_generated_ids = output[0]
                
                generated_text = self.processor.tokenizer.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
                
                candidates.append({
                    'input_ids': new_generated_ids.unsqueeze(0),
                    'generated_text': generated_text,
                })
            
            while(len(candidates) > 1):
                # randomly select two candidates
                candidate1 = candidates.pop(np.random.randint(len(candidates)))
                candidate2 = candidates.pop(np.random.randint(len(candidates)))
                outputs = [candidate1['generated_text'], candidate2['generated_text']]
                best_index = self.judge(image, prompt, outputs, type=stage[1:-1].lower())
                if best_index == 0:
                    candidates.append(candidate1)
                else:
                    candidates.append(candidate2)
            
            input_ids = candidates[0]['input_ids']

        final_output = self.processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True)
        return final_output
    
    def generate_inner(self, message, dataset=None):
        type = "beam"
        if type == "all":
            return self.generate_inner_best_of_N(message, dataset)
        elif type == "sentence":
            return self.generate_inner_sentence_beam(message, dataset)
        else:
            return self.generate_inner_stage_beam(message, dataset)