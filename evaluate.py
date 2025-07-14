from transformers import AutoTokenizer, AutoModelForCausalLM
import bert_score
import torch

user_input = "I dont know what to do I dont know how to proceed I'm lost!"
emotion_label = "confusion"
suicide_risk = "Not on Suicide Watch"
chatbot_response = '''I can see that you're feeling confused right now, and it's okay to feel that way. Confusion is a natural part of life, and it often means we are faced with something new or challenging. It might help to break things down into smaller steps or seek guidance from someone who has experience with the situation at hand. Remember, there's no need to rush or figure everything out on your own. You're not alone in this, and support is available if you need it. If you're in Germany, you can reach out to TelefonSeelsorge anonymously and for free at 0800 111 0 111. They are there to listen and help when things seem overwhelming.'''
reference_response = "I know its confusing right now but trust me it will all make sense later. I know its overwhelming but its natural dont worry about it. It might help to break things down into smaller steps or seek guidance from someone who has experience with the situation at hand"


P, R, F1 = bert_score.score([chatbot_response], [reference_response], lang="en", model_type="bert-base-uncased")
print(f"\nBERTScore:\nPrecision: {P.item():.4f}, Recall: {R.item():.4f}, F1: {F1.item():.4f}")

# 2. Evaluate using LLM-as-a-Judge
# model_name = "tiiuae/falcon-7b-instruct"
# judge_tokenizer = AutoTokenizer.from_pretrained(model_name)
# judge_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
# judge_model.eval()


prompt = f"""You are a helpful assistant evaluating an emotional support chatbot.
Context:
User message: "{user_input}"
Detected Emotion: {emotion_label}
Suicide Risk Level: {suicide_risk}
Chatbot Response: "{chatbot_response}"

Evaluate the response for empathy, helpfulness, and appropriateness for mental health support. 
Give a 1-line rating out of 10 and justify it briefly."""

# # Generate judgment
# inputs = judge_tokenizer(prompt, return_tensors="pt", truncation=True)
# with torch.no_grad():
#     outputs = judge_model.generate(**inputs, max_length=512)

# result = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("\nLLM-as-a-Judge Evaluation:\n", result)
