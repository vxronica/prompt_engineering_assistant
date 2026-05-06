# chatbot.py

# imports

# re is used to clean text. e.g:removing punctuation
# get_close_matches is used to fix small spelling mistakes in user questions
# numpy is used for vector math and ranking similarity scores
# torch is used by the Hugging Face model during text generation
# SentenceTransformer turns text into embeddings for semantic search
import re
from difflib import get_close_matches
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
# These load the FLAN-T5 tokenizer and generation model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# knowledge base
# short chunks work better for RAG because they are easier to match to user questions.
knowledge_base = [

    # basic Definitions
    "A prompt is the input, instruction, question, or task given to an AI model to guide its response.",
    "Prompt engineering is the process of developing and improving prompts to get better responses from large language models.",
    "A large language model, or LLM, is an AI system trained on large amounts of text to understand and generate language.",
    "A token is a unit of language that a model uses to process text. A token can be a word, part of a word, or punctuation.",
    "An AI hallucination is when a language model confidently generates false, incorrect, or nonsensical information.",
    "Retrieval augmented generation, or RAG, is a technique where a model uses retrieved external knowledge to answer questions.",
    "Chain-of-thought prompting asks the model to reason through a problem step by step.",
    "Zero-shot prompting means asking a model to complete a task without giving examples.",
    "Few-shot prompting means giving a model one or more examples before asking it to complete a task.",
    "One-shot prompting is a type of few-shot prompting that gives the model exactly one example.",
    "Meta prompting focuses on the structure and syntax of a task rather than the content.",
    "Self-consistency samples multiple reasoning paths and chooses the most consistent answer.",
    "Tree of Thoughts explores multiple reasoning paths before choosing an answer.",
    "Automatic prompt engineering is when a model generates and evaluates its own prompts.",
    "Prompt chaining breaks a large task into smaller subtasks.",
    "Generate knowledge prompting adds useful knowledge before answering to improve accuracy.",
    "ReAct prompting combines reasoning with actions such as retrieving information or using tools.",
    "Reflexion uses feedback to help a model evaluate and improve its previous answer.",
    "Program-aided language models, or PAL, use programming tools such as Python as part of the reasoning process.",
    "Multi-persona prompting asks the model to consider a problem from multiple roles or perspectives.",
    "Temperature controls how deterministic or creative a model response is.",
    "Top-p controls how much variety the model can use when choosing possible next tokens.",
    "A stop sequence is a string that tells the model when to stop generating.",
    "A frequency penalty reduces repeated wording by discouraging tokens that have already appeared often.",
    "A presence penalty discourages repeated tokens regardless of how many times they appeared.",
    "Max length limits the number of tokens in a model response.",
    "Tokens are important because language models read, generate, and limit text based on tokens rather than full words.",

    # prompt Writing
    "Good prompts usually include clear instructions, relevant context, input data, and a desired output format.",
    "To write better prompts, include a specific task, useful context, constraints, examples, and the desired output format.",
    "To build better prompts, start with a simple prompt, test the response, and revise the prompt based on what is missing or unclear.",
    "A good prompt should be specific and direct instead of vague or unclear.",
    "Adding examples to a prompt can help the model understand the desired style, structure, or format.",
    "Instructions tell the model what task to perform, such as write, summarize, classify, or explain.",
    "Context gives the model background information that helps it produce a more relevant answer.",
    "Input data is the specific question, text, or material the user wants the model to respond to.",
    "Output indicators tell the model what format the answer should follow, such as a list, paragraph, table, or summary.",
    "Tell the model what to do rather than focusing only on what not to do.",
    "If a task is large, break it into smaller subtasks so the model can handle each part more clearly.",
    "Subtasks help because they break large prompts into smaller and clearer steps that are easier for the model to follow.",
    "Designing prompts is an iterative process, so users should start simple and adjust the prompt based on the output.",
    "Prompt engineering helps users communicate more effectively with AI systems.",

    # hallucinations
    "AI hallucinations can happen when a model lacks reliable context or guesses missing information.",
    "AI hallucinations can also happen when generated text sounds correct but is not factual.",
    "AI hallucinations often happen when a model tries to predict likely text without enough factual grounding.",
    "To reduce hallucinations, provide reliable reference material for the model to use.",
    "To reduce hallucinations, ask the model to cite sources or explain where information came from.",
    "To reduce hallucinations, give the model permission to say 'I don't know' when it is unsure.",
    "When a model is unsure of a response, it should say it does not know instead of guessing.",
    "If a model is unsure of a response, it may generate unreliable or hallucinated information.",
    "To reduce hallucinations, ask the model to review its own answer for possible inaccuracies.",
    "To reduce incorrect responses, provide reliable context, use RAG, ask for sources, and write more specific prompts.",
    "Running the same prompt multiple times and comparing outputs can help reveal possible hallucinations.",
    "If the model gives very different answers to the same prompt, the answer may be unreliable.",
    "Asking the model to state its confidence level can help prevent it from presenting guesses as facts.",
    "A useful uncertainty prompt is: If you are uncertain about any part of your answer, state your confidence level.",
    "RAG can reduce hallucinations because the model has relevant context to use when generating an answer.",
    "Providing reference material gives the model a factual base to use when answering.",
    "Hallucinations are bad because they can make users trust false, misleading, or unsupported information.",
 
    # RAG
    "RAG helps ground model responses in specific reference material instead of relying only on the model's internal training.",
    "RAG stands for retrieval augmented generation, a technique where a model retrieves relevant information and uses it to generate a more grounded answer.",
    "RAG is useful for factual or knowledge-intensive questions.",
    "A RAG chatbot retrieves relevant knowledge base chunks before generating a response.",
    "In a RAG pipeline, the user question is embedded, compared to knowledge base embeddings, and matched with the most relevant contexts.",
    "After retrieval, the selected contexts are passed to the language model so it can generate a grounded answer.",
    "Semantic retrieval uses embeddings to find text with similar meaning instead of exact keyword matches.",
    "Semantic retrieval is a search technique that finds information based on meaning and context rather than exact keyword matches.",
    "Embeddings are numerical vector representations of text used for semantic similarity comparisons.",
    "Cosine similarity measures how similar two embedding vectors are.",
    "SentenceTransformers can generate embeddings for semantic search and retrieval tasks.",
    "Vectors are lists of numbers that represent information, such as text, in a format computers can compare.",

    # citations and sources
    "Asking the AI to provide citations can make its response easier to verify.",
    "A good citation prompt is: Answer the question and provide a citation or source for each key point.",
    "If the model is unsure of a source, it should say so instead of inventing one.",
    "Citations help users double-check whether an AI response is accurate.",
    "A useful reference-based prompt is: Based on this document, please answer the question.",

    # chain-of-thought
    "Chain-of-thought prompting is useful for multi-step problems, math, logic, troubleshooting, and complex reasoning.",
    "Chain-of-thought prompting can reduce logic gaps by encouraging intermediate reasoning steps.",
    "A simple chain-of-thought prompt is: Think step by step before answering.",
    "Chain-of-thought prompting is usually better than few-shot prompting for complex reasoning tasks.",
    "Chain-of-thought prompting is useful when the task requires step-by-step reasoning.",
    "Chain-of-thought reasoning means solving a problem through intermediate reasoning steps before giving the final answer.",

    # few-Shot / zero-Shot
    "Zero-shot prompting can work well for simple tasks because language models are trained on large amounts of text.",
    "Few-shot prompting is useful when the user wants to guide the model's style, structure, or format.",
    "Few-shot prompting may work better than zero-shot prompting when the desired output format is specific.",
    "One-shot prompting may be enough for simple tasks, while several examples may be better for more complex formatting needs.",
    "Few-shot prompting does not work as well for complex reasoning tasks such as advanced arithmetic or symbolic reasoning.",

    # comparisons
    "Zero-shot prompting uses no examples, while few-shot prompting gives the model one or more examples.",
    "One-shot prompting gives one example, while few-shot prompting can give multiple examples.",
    "Few-shot prompting is usually better than one-shot prompting when the task needs stronger guidance.",
    "One-shot prompting may be better than few-shot prompting when the task is simple and only needs one example.",
    "Chain-of-thought prompting is better than few-shot prompting when a task requires step-by-step reasoning.",
    "RAG is better than ordinary prompting when the answer needs specific reference material or factual grounding.",
    "There is no single best prompting technique because the best method depends on the task.",
    "Chain-of-thought prompting is useful for complex reasoning tasks, while few-shot prompting is useful for showing examples of the desired format.",
    "RAG is useful when the answer needs factual grounding from reference material.",
    "Few-shot prompting is usually better than one-shot prompting when the task needs stronger guidance, but one-shot prompting may be enough for simple tasks.",
    "Chain-of-thought prompting is usually better than one-shot or few-shot prompting when the task requires step-by-step reasoning.",

    # Advanced prompting techniques
    "Meta prompting can be useful for complex reasoning tasks such as mathematical problem-solving or coding challenges.",
    "Prompt chaining is useful for conversational assistants, debugging, and answering questions from long texts.",
    "In prompt chaining, the output from one step is used as input for the next step.",
    "Tree of Thoughts can explore more possibilities than a single linear chain-of-thought response.",
    "Automatic prompt engineering can help discover better prompts than a human-written prompt.",
    "Active-prompt selects uncertain questions for human annotation after generating multiple responses.",
    "Directional stimulus prompting uses guidance to steer the model toward a desired summary.",
    "Multimodal chain-of-thought prompting allows models to reason using multiple types of data, such as text and images.",
    "Multi-persona prompting can reveal blind spots and reduce one-dimensional thinking.",
    "An example of multi-persona prompting is asking the model to answer as a scientist, an ethicist, and a student.",

    # model settings
    "Lower temperature values usually produce more focused and factual responses.",
    "Higher temperature values usually produce more creative but less predictable responses.",
    "Temperature affects output by controlling randomness: lower temperature makes responses more focused and factual, while higher temperature makes responses more creative and less predictable.",
    "Lower top-p values usually produce more focused responses.",
    "Higher top-p values usually allow more variety and creativity.",
    "Max length can prevent responses from becoming too long and can help control cost.",
    "Stop sequences can help control response length and structure.",
    "Higher presence penalty values can make text more diverse, while lower values help the model stay focused.",
    "Users should usually avoid changing temperature and top-p at the same time.",
    "Users should usually avoid changing frequency penalty and presence penalty at the same time.",
    "AI response results can vary depending on the model and model version used.",

    # chatbot purpose
    "This chatbot teaches prompt engineering in a friendly, supportive, and slightly academic tone.",
    "This chatbot is designed for college students, AI beginners, and everyday users who want better AI responses.",
    "This chatbot gives concise educational answers about prompting, hallucinations, and AI response quality.",
    "What is this chatbot for? This chatbot teaches prompt engineering, hallucination reduction, RAG, citations, model settings, and AI response quality.",
    "This chatbot can provide information about prompt engineering, hallucinations, RAG, citations, embeddings, tokens, and model settings."
]


# Embedding Model

# Load the embedding model which turns each text chunk into a vector
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# this function normalizes vectors so cosine similarity works correctly
def normalize(vectors):
    # calculate the length of each vector
    # tiny number prevents division by zero
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

    # fivide each vector by its length
    return vectors / norms


# convert all knowledge base chunks into embeddings
kb_vectors = embedder.encode(knowledge_base, convert_to_numpy=True)

# normalize the knowledge base embeddings.
kb_vectors = normalize(kb_vectors)

# generation Model

# this is the model used to generate the final answer
MODEL_NAME = "google/flan-t5-small"

# the tokenizer converts normal text into tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# the model generates text from the retrieved context
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# this function sends a prompt to FLAN-T5 and returns the generated answer
def generate_answer(prompt, max_new_tokens=100):
    # convert the prompt into tokens
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # no_grad means we are not training the model, only using it.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,

            # limits how long the answer can be
            max_new_tokens=max_new_tokens,

            # makes output more consistent and less random
            do_sample=False,

            # greedy decoding chooses the most likely next token each time
            num_beams=1
        )

    # decode the model output back into readable text
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()



# text cleaning and query helpers

# cleans user text before searching
def clean_text(text):
    # lowercase makes matching easier
    text = text.lower()

    # remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # remove spaces at the beginning or end
    return text.strip()


# corrects small typos in important prompt engineering terms
def correct_query_terms(text):
    # these are important words the chatbot should recognize
    vocabulary = [
        "prompt", "prompts", "prompting", "engineering",
        "hallucination", "hallucinations",
        "rag", "retrieval", "generation",
        "token", "tokens",
        "temperature", "top",
        "few", "shot", "zero", "one",
        "chain", "thought", "reasoning",
        "llm", "model", "context",
        "reduce", "better", "write", "build",
        "best", "technique", "techniques",
        "cause", "causes",
        "citation", "citations", "source", "sources",
        "frequency", "presence", "penalty",
        "length", "stop", "sequence",
        "meta", "automatic", "persona",
        "factual", "incorrect", "response", "responses",
        "subtask", "subtasks", "chatbot"
    ]

    corrected_words = []

    # check each word in the user question
    for word in text.split():
        # find the closest matching known word
        match = get_close_matches(word, vocabulary, n=1, cutoff=0.75)

        # use the corrected word if a close match is found
        if match:
            corrected_words.append(match[0])
        else:
            corrected_words.append(word)

    # rebuild the corrected question
    return " ".join(corrected_words)


# removes common words that do not help retrieval
def remove_stopwords(text):
    stopwords = {
        "what", "is", "a", "an", "the", "how", "can", "i",
        "does", "do", "are", "to", "of", "and", "in", "on",
        "for", "me", "my", "about", "tell", "more", "please"
    }

    # keep only useful words
    return " ".join(word for word in text.split() if word not in stopwords)



# retrieval Scoring Functions

# measures exact keyword overlap between the user query and a KB chunk
def keyword_overlap_score(query, text):
    # unique words from the user query
    q_words = set(query.split())

    # unique words from the KB chunk
    t_words = set(clean_text(text).split())

    # avoid division by zero
    if len(q_words) == 0:
        return 0

    # return the percentage of query words that appear in the chunk
    return len(q_words & t_words) / len(q_words)


# measures whether the important topic words match
def topic_match_score(search_q, text):
    query_words = set(search_q.split())
    text_words = set(clean_text(text).split())

    # rhese are the important domain words for the chatbot
    important_words = {
        "prompt", "prompts", "prompting", "engineering",
        "llm", "token", "tokens",
        "hallucination", "hallucinations",
        "temperature", "top", "rag",
        "few", "shot", "zero", "one",
        "chain", "thought", "reasoning",
        "best", "technique", "techniques",
        "build", "better",
        "citation", "citations", "source", "sources",
        "frequency", "presence", "penalty",
        "length", "stop", "sequence",
        "meta", "automatic", "persona",
         "factual", "incorrect", "response", "responses",
        "subtask", "subtasks", "chatbot"
    }

    # get only important topic words from the query
    query_topics = query_words & important_words

    # if there are no important topic words, return 0
    if len(query_topics) == 0:
        return 0

    # return how many important query topics appear in the KB chunk
    return len(query_topics & text_words) / len(query_topics)


# detects what kind of question the user asked
def detect_intent(question):
    q = question.lower()

    # comparison questions should be checked first
    if (
        "difference" in q
        or "compare" in q
        or "which is better" in q
        or "better than" in q
        or "better" in q
    ):
        return "comparison"

    # recommendation questions
    elif (
        "best" in q
        or "should i use" in q
        or "what type" in q
    ):
        return "recommendation"

    # definition questions
    elif q.startswith("what is") or q.startswith("what are") or q.startswith("what does") or q.startswith("define"):
        return "definition"

    # how-to questions
    elif q.startswith("how") or "how to" in q:
        return "how_to"

    # cause/explanation questions
    elif q.startswith("why") or "what causes" in q:
        return "explanation"

    # default category
    else:
        return "general"


# gives extra score to chunks that match the question type
def intent_match_score(intent, text):
    text_lower = text.lower()

    # definition chunks usually contain "is" or "means."
    if intent == "definition":
        if " is " in text_lower or "means" in text_lower:
            return 1.0

    # how-to chunks often begin with "To..." or use action words
    elif intent == "how_to":
        if (
            text_lower.startswith("to ")
            or "include" in text_lower
            or "provide" in text_lower
            or "ask" in text_lower
            or "use" in text_lower
            or "helps" in text_lower
            or "can help" in text_lower
        ):
            return 1.0

    # explanation chunks often explain causes
    elif intent == "explanation":
        if "happen" in text_lower or "when" in text_lower or "because" in text_lower:
            return 1.0

    # comparison chunks compare two techniques
    elif intent == "comparison":
        if "while" in text_lower or "better than" in text_lower or "usually better" in text_lower:
            return 1.0

    # recommendation chunks answer "best technique" style questions
    elif intent == "recommendation":
        if "no single best" in text_lower or "depends on the task" in text_lower or "useful for" in text_lower or "for factual information" in text_lower:
            return 1.0

    # no intent match
    return 0.0


# combines all scoring methods to rank KB chunks
def compute_scores(search_q, clean_q):
    # embed the cleaned user query
    q_vec = embedder.encode([search_q], convert_to_numpy=True)

    # normalize query embedding
    q_vec = normalize(q_vec)

    # semantic score compares query embedding to every KB embedding
    semantic_scores = (q_vec @ kb_vectors.T).flatten()

    # keyword score checks exact word overlap
    keyword_scores = np.array([
        keyword_overlap_score(search_q, kb) for kb in knowledge_base
    ])

    # topic score checks important domain words
    topic_scores = np.array([
        topic_match_score(search_q, kb) for kb in knowledge_base
    ])

    # detect question type
    intent = detect_intent(clean_q)

    # intent score checks if KB chunks match the question type
    intent_scores = np.array([
        intent_match_score(intent, kb) for kb in knowledge_base
    ])

    # final weighted score
    # semantic similarity is the largest part, but keyword/topic/intent help avoid weird matches
    scores = (
        0.45 * semantic_scores +
        0.25 * keyword_scores +
        0.20 * topic_scores +
        0.10 * intent_scores
    )

    return scores


# checks whether the generated answer still mentions the user's main topic
def answer_mentions_topic(answer, search_q):
    answer_words = set(clean_text(answer).split())
    query_words = set(search_q.split())

    important_words = {
        "prompt", "prompts", "prompting", "engineering",
        "hallucination", "hallucinations",
        "rag", "token", "tokens",
        "few", "shot", "zero", "one",
        "chain", "thought", "reasoning",
        "best", "technique", "techniques",
        "build", "better",
        "citation", "citations", "source", "sources",
        "frequency", "presence", "penalty",
        "length", "stop", "sequence",
        "meta", "automatic", "persona",
        "factual", "incorrect", "response", "responses",
        "subtask", "subtasks", "chatbot"
    }

    # important words from the user's query
    query_topics = query_words & important_words

    # if there are no topic words, do not block the answer
    if len(query_topics) == 0:
        return True

    # return True if the answer mentions at least one important topic word
    return len(query_topics & answer_words) > 0


# Main Chatbot Function

# This function runs the full RAG pipeline
def ask_chatbot(question, top_k=3, conf_threshold=0.28, debug=False):

    # step 1: Clean the user question
    clean_q = clean_text(question)

    # step 2: Correct small spelling mistakes
    clean_q = correct_query_terms(clean_q)

    # handle greetings
    greetings = ["hi", "hello", "hey", "hi there", "hey there"]

    if clean_q in greetings:
        return "Hi! 👋 I'm your prompt engineering assistant."

    # handle empty input
    if clean_q == "":
        return "Please enter a question about prompt engineering or AI hallucinations."

    # handle obvious out-of-scope questions
    out_of_scope_terms = [
        "weather",
        "sports",
        "stock",
        "english paper",
        "climate science",
        "vector calculus",
        "baseball",
        "api"
    ]

    if any(term in clean_q for term in out_of_scope_terms):
        return "I don't know based on my knowledge base."
    #handling math related queiries
    if "math" in clean_q or "mathematics" in clean_q:
        return "For math problems, chain-of-thought prompting can help because it asks the model to work through the problem step by step."

    # step 3: remove stopwords for retrieval
    search_q = remove_stopwords(clean_q)

    if search_q == "":
        return "I don't know based on my knowledge base."

    # step 4: score all knowledge base chunks
    scores = compute_scores(search_q, clean_q)

    # step 5: retrieve top-k chunks
    top_indices = np.argsort(scores)[::-1][:top_k]

    # highest retrieval score.
    top_score = float(scores[top_indices[0]])

    # Optional debug mode shows what the system retrieved
    if debug:
        print("Original question:", question)
        print("Cleaned question:", clean_q)
        print("Search question:", search_q)
        print("Intent:", detect_intent(clean_q))
        print("Top score:", top_score)
        print("Top matches:")

        for i in top_indices:
            print("-", knowledge_base[i])

    # step 6: If retrieval confidence is too low, do not guess
    if top_score < conf_threshold:
        return "I don't know based on my knowledge base."

    # step 7: build retrieved context
    # this is the "Augmented" part of Retrieval-Augmented Generation
    context = "\n".join(
        [f"Context {rank + 1}: {knowledge_base[i]}" for rank, i in enumerate(top_indices)]
    )

    # step 8: build the generation prompt
    # The model is instructed to answer only from retrieved context.
    prompt = (
        "You are a friendly prompt engineering tutor.\n"
        "Answer the user's question using ONLY the contexts below.\n"
        "Prefer Context 1 unless another context directly answers better.\n"
        "Do not use outside knowledge.\n"
        "If the contexts do not answer the question, say: I don't know based on my knowledge base.\n"
        "Write one concise complete sentence.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    # step 9: Generate answer using FLAN-T5
    answer = generate_answer(prompt)

    # step 10: If the model gives a very short answer, return the best retrieved chunk
    if len(answer.split()) < 5:
        return knowledge_base[top_indices[0]]

    # step 11: If the answer drifts off-topic, return the best retrieved chunk
    if not answer_mentions_topic(answer, search_q):
        return knowledge_base[top_indices[0]]

    # final answer
    return answer