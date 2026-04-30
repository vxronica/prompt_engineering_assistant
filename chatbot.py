# chatbot.py

import re
from difflib import SequenceMatcher

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Knowledge Base
# -----------------------------
knowledge_base = [
    "AI hallucinations happen when language models confidently generate false or nonsensical information.",
    "A good type of prompting is chain-of-thought prompting, where you instruct the model to reason out an answer step-by-step. This helps reduce logic gaps and makes it harder for the model to hallucinate mid prompt.",
    "Some simple techniques that help make a good prompt are using Chain-of-Thought, using reference material such as RAG, asking the AI to provide sources, and being very specific about your expectations for what you want the output to look like.",
    "Chain-of-thought prompting can be used for multi-step problems, math, logic problems, complex problems, and troubleshooting problems. It is useful when a step-by-step approach is best.",
    "Providing AI with extra material to pad its knowledge base gives the AI a factual base to work from. Using a prompt such as 'Based on this document, please answer...' helps reduce hallucinations.",
    "When AI is required to cite its source, it tends to give evidence from recognized sources. This also gives the user a way to double check the answer.",
    "A good prompt structure for a cited answer could be: 'Answer the following question and for each key point, provide a citation or source. If you're unsure of a source, indicate that.' This helps create transparency in the AI's response.",
    "Sometimes explicitly telling AI to communicate its confidence level and acknowledge limitations is helpful. A phrase such as 'If you're uncertain about any part of your answer, state your confidence level' prevents AI from presenting guesses as facts.",
    "Multi-persona prompting is when you ask the AI to consider a problem from multiple perspectives or different roles. This helps create internal checks and balances, reveals blind spots, and reduces one-dimensional thinking that can lead to hallucinations.",
    "An effective multi-persona prompt might be: 'Analyze this decision from three perspectives: 1) As a CFO focused on financial risk, 2) As the head of R&D focused on innovation, and 3) As a customer experience manager focused on user experience. Then using these viewpoints, produce a balanced recommendation.'",
    "For maximum effectiveness, use a layered approach: start with reference materials to ground the AI, then use Chain-of-Thought to work through the problem, and finally ask for citations to verify key claims.",
    "Common mistakes in prompt engineering include being too vague, not specifying context, not providing examples of the desired output format, failing to specify the AI's persona, accepting the first answer without iteration, and not instructing the AI on how to handle uncertainty.",
    "The more detailed your prompt, the more likely the AI response will be useful. A well-constructed prompt typically includes clear context, a specific task or question, constraints or format requirements, reference materials if possible, and instructions for handling uncertainty.",
    "Running the same prompt multiple times and comparing the output can help reveal hallucinations if the information varies greatly.",
    "Giving the AI permission to say 'I don't know' can be very useful in getting correct information. You can also ask the AI to review its own work for inaccuracies.",
    "Prompt engineering is the process of developing and improving prompts to better understand and use large language models.",
    "Prompt engineering involves designing techniques that help models interact with tasks more effectively.",
    "When designing and testing prompts, users typically interact with a model through an API. Adjusting the settings for the model can change how responses are generated. Responses can become more factual or more creative depending on the temperature or top-p settings.",
    "Temperature adjusts how deterministic the responses are. Lower temperature helps produce more factual results. Higher temperature helps produce more creative but less factual responses.",
    "Top-p is a sampling technique used with temperature. Lower top-p values tend to produce more exact and factual responses. Higher top-p values allow more variety and creativity.",
    "Max length manages the number of tokens in the model response to prevent answers from being too long. It also helps control costs.",
    "A stop sequence is a string that stops the model from continuing to generate tokens. Stop sequences can help control output length and output structure.",
    "A frequency penalty can be applied to control how many times a token appears in the response. A higher frequency penalty decreases repeated wording.",
    "A presence penalty applies a penalty on repeated tokens regardless of how many times they repeat. Increasing this value can lead to more diverse and creative text, while lower values help the language model stay focused.",
    "Do not modify both temperature and top-p at the same time. Do not modify both frequency and presence penalty at the same time.",
    "Response results can vary depending on which model and which version of the model is used.",
    "Some common elements of prompts include instructions, context, input data, and output indicators.",
    "Instructions tell the language model to perform a specific task. Context gives the model additional external information to guide better responses. Input data is the question or content the user wants processed. Output indicators tell the model what type or format of output is desired.",
    "Designing prompts is an iterative process, so users should start with simple prompts and gradually adjust them to get better outputs. If the task is large, it should be broken into smaller subtasks.",
    "Clearly indicate the instructions for the model such as write, summarize, or classify at the beginning of the prompt. Also be specific by including relevant details.",
    "Do not include too many details because there is a limit to the number of input tokens. The prompt should still be specific and direct. Avoid imprecise descriptions such as 'a few' rather than a specific number or range.",
    "Tell the model what to do rather than focusing on what not to do.",
    "Zero-shot prompting is a simple prompting technique that does not provide any examples. It can work well because models are trained on large amounts of data.",
    "If a zero-shot prompt does not provide the desired response, use a few-shot prompt.",
    "Few-shot prompting is a simple prompting technique that provides one or more examples. It is helpful for guiding output style or format.",
    "Simple tasks can use one example in few-shot prompting, while more complex tasks may require more examples such as 3-shot or 5-shot.",
    "Few-shot prompting does not work as well with complex reasoning tasks such as advanced arithmetic, commonsense reasoning, and symbolic reasoning. Chain-of-thought prompting is better for these tasks.",
    "Chain-of-thought prompting is an advanced prompting technique that allows the language model to perform complex reasoning by using intermediate reasoning steps.",
    "An example of a zero-shot chain-of-thought prompt includes an instruction such as 'think step-by-step.'",
    "Meta prompting is an advanced prompting technique that focuses on the structure and syntax of a task rather than its content.",
    "Some characteristics of meta prompting are that it is structure-oriented, syntax-focused, uses abstract examples, is versatile, and uses a categorical approach.",
    "Meta prompting has advantages over few-shot prompting including token efficiency, fairer comparisons, and zero-shot efficiency not influenced by specific examples.",
    "Meta prompting works well for complex reasoning tasks such as mathematical problem-solving and coding challenges.",
    "Self-consistency is a very advanced prompting technique that samples multiple reasoning paths through few-shot chain-of-thought and then selects the most consistent answer.",
    "Generate knowledge prompting is a prompting technique that incorporates knowledge for the model to increase the accuracy of its predictions.",
    "Prompt chaining is a prompting technique that improves reliability and performance by breaking large tasks into smaller subtasks.",
    "In prompt chaining, the output from one step is used as input for the next step.",
    "Prompt chaining is helpful for conversational assistants, debugging problems, and answering questions using quotes from large texts.",
    "A prompt chaining example for long text question answering is to first extract quotes from the text and then use those quotes in the next step to answer the question.",
    "Tree of Thoughts is a prompting technique that uses tree search ideas to explore multiple reasoning paths and determine a better response.",
    "Chain-of-thought and self-consistency are more linear thought processes, while Tree of Thoughts can explore more possible paths.",
    "Retrieval augmented generation, or RAG, is a prompting technique used for complex and knowledge-intensive tasks. It allows a language model to access external sources for its knowledge base.",
    "RAG is useful for factual information that can change over time without having to retrain the entire model.",
    "Automatic reasoning and tool-use, or ART, combines chain-of-thought reasoning with the use of tool libraries.",
    "ART allows a language model to respond to a new task, choose a multi-step reasoning process, and call external tools to integrate into its output.",
    "Automatic prompt engineering is a prompting technique where a model generates and evaluates its own prompts.",
    "Automatic prompt engineering can help discover a better zero-shot chain-of-thought prompt than a human-written prompt.",
    "Active-prompt is a prompting technique where a model generates multiple responses and each response is given an uncertainty metric. The most uncertain questions are selected to be annotated by a human.",
    "Directional stimulus prompting is a prompting technique that uses a tunable method to guide the model toward generating a desired summary.",
    "Program-aided language models, or PAL, use programming tools such as Python as an intermediate reasoning step. This can improve accuracy compared to using only text reasoning.",
    "ReAct prompting combines reasoning with actions like retrieving information to help models handle exceptions and produce more reliable answers.",
    "Reflexion uses linguistic feedback to help models learn from previous mistakes by generating an answer, evaluating it, and improving it based on feedback.",
    "Multimodal chain-of-thought prompting allows models to reason using multiple types of data such as text and images.",
    "An AI token is a unit of language that an LLM uses for language processing. A token can be a whole word, part of a word, or punctation. LLMs use tokens to process input, predict a sequence of tokens that are most likely correct, then converts the tokens back to words that can be read and understood."
]


# -----------------------------
# Embedding Model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True)
kb_vectors = normalize(kb_vectors)


# -----------------------------
# Generation Model
# -----------------------------
MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_answer(prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# -----------------------------
# Utility Functions
# -----------------------------
def fuzzy_match(text, target, threshold=0.70):
    return SequenceMatcher(None, text, target).ratio() >= threshold


# -----------------------------
# Main Chatbot Function
# -----------------------------
def ask_chatbot(question, top_k=4, conf_threshold=0.35, debug=False):

    # Clean input
    clean_q = re.sub(r"[^\w\s]", "", question.lower()).strip()

    # Greeting handling
    greetings = ["hi", "hello", "hey", "hi there", "hey there"]
    if clean_q in greetings:
        return "Hi! 👋 I’m your prompt engineering assistant. Ask me anything about prompt engineering!"

    # Out-of-scope filter
    out_of_scope_terms = ["cs335", "cs355", "weather", "sports", "stock"]
    if any(term in clean_q for term in out_of_scope_terms):
        return "I don't know based on my knowledge base."

    # Handle common question explicitly
    if (
        fuzzy_match(clean_q, "what is prompt engineering") or
        fuzzy_match(clean_q, "explain what prompt engineering is")
    ):
        return "Prompt engineering is the process of developing and improving prompts to better understand and use large language models."

    # Normalize token question
    if fuzzy_match(clean_q, "what is a token") or fuzzy_match(clean_q, "what is an ai token"):
        clean_q = "what is a token"

    # Embed question
    q_vec = embedder.encode([clean_q], convert_to_numpy=True)
    q_vec = normalize(q_vec)

    # Similarity scoring
    scores = (q_vec @ kb_vectors.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_score = float(scores[top_indices[0]])

    if debug:
        print("Cleaned:", clean_q)
        print("Top score:", top_score)

    # Confidence check
    if top_score < conf_threshold:
        return "I don't know based on my knowledge base."

    # Build context
    context = "\n- " + "\n- ".join([knowledge_base[i] for i in top_indices])

    # Prompt
    prompt = (
        "You are a helpful chatbot.\n"
        "Answer ONLY using the context.\n"
        "If unsure, say: I don't know based on my knowledge base.\n\n"
        f"Context:{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    return generate_answer(prompt)