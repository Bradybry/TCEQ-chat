# TCEQ-Chat

This repo includes a simple gradio chatbot wrapper that allows users to chat with Claude-v1.3 leveraging OpenAI embeddings of Texas Commission on Environmental Quality (TCEQ) to supply regulatory rule search, synthesis, and references.

## How to use
Make sure you have installed required packages - Anthropic, OpenAI, Langchain, Gradio, pandas, and scipy

Edit the rule_path variable in the TCEQ_chat.py script and then run it. You should be able to chat with your regulations. Currently I do not handle token excedance so conversations must be limited to the ~8500 tokens. Conversations can be saved to .pkls which is a list of tuples of questions and repsonses. The context message with relevant rules is not preserved. 

Currently this is a very minimal implentation. I will be working on adding functionality, improving prompts/embeddings/logging, and adding some error handling. However, I have found it remarkably useful to figure out relevant rules that I am familar with but do not have memorized. 

Hallucinations seem to be pretty minimal, however it does appear to misintepret rules occasionally. In addition, it almost always cite's the corresponding rules so you can and SHOULD always check the actual rule before using the responses to make any major decisoins. However, if you are already familiar with the rules and are just looking for a fast way to search and cross reference multiple rules at one time, this tool is incredibly useful.

I intend to build out the feaures a little more to allow users to get the corresponding snippets of the rules they are referncing that can then be confirmed by the engineer or used in the next step of their proccess. In addition a rule set selector allowing users to switch between rule files will be added so that the system doesn't have to be restarted just to reference other rules.

I will be adding additional rule sets overtime. Mostly on an as need basis for my own work. However, if you find this tool useful but want a specific chapter included, just let me know and I can get it done!One day I hope to include other standards and regulations, however I may not be able to publish some of that due to licensing of the standards.

Finally, DO NOT use this if you do not already have some familarity with TCEQ rules. And do not trust the output at face value. LLMs are not reliable and I have not done extensive testing to see where it struggles. But if you know what you are doing, I think you will find this tool incredibly useful.

## rule extractor.ipynb

This is the notebook that I used to extract the rules into dataframes, get embeddings, and save to a pkl for the chatbot to use. Rules were downloaded in pdf from the TCEQ website and converted to txt via blubeam. Each pdf required slighly customized filters but overall minimal editing is required to apply the same technique to other TCEQ rulesets. 