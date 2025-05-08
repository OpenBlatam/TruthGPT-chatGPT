To implement reasoning with an older or less capable LLM in LlamaIndex, you can use prompt engineering to guide the model step-by-step, decompose complex queries into sub-questions, or use agent workflows that explicitly structure the reasoning process. LlamaIndex supports techniques like multi-step queries, sub-question decomposition, and custom prompt templates to help older models perform better at reasoning tasks, even if their native capabilities are limited. For example, you can use the SubQuestionQueryEngine or MultiStepQueryEngine to break down reasoning into smaller, more manageable steps for the model to handle, and customize prompts to be as explicit as possible about the reasoning process required (LlamaIndex Prompt Engineering and LLM Customization, Multi Document Queries).

Sources:

Multi Document Queries
SubQuestionQueryEngine

