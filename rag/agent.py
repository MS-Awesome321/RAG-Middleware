from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TextIteratorStreamer
from threading import Thread

name = "Qwen/Qwen3-0.6B"
config = AutoConfig.from_pretrained(name, tie_word_embeddings=False)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    device_map="auto",
    config = config
)
device = 'mps'

def response_generator(chat):
    '''
    Takes chat dict as input. Example:

    chat = [
        { "role": "user", "content": "Write a haiku"},
        etc...
    ]
    '''

    question = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    question = tokenizer(question, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    generation_kwargs = dict(
        question, 
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id, 
        max_new_tokens=2048, 
        temperature=0.1,
        repetition_penalty=1.25
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for output in streamer:
        output = output.replace("<|im_end|>", "")
        output = output.replace("<think>", "**Thinking**:")
        output = output.replace("</think>", "\n**Done Thinking.**\n")
        yield output


if __name__=='__main__':

    prompt = "Write a haiku about Retrieval Augmented Generation."

    chat = [
        { "role": "user", "content": prompt},
    ]
    question = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    question = tokenizer(question, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    generation_kwargs = dict(
        question, 
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id, 
        max_new_tokens=2048, 
        temperature=0.1,
        repetition_penalty=1.25
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for output in streamer:
        print(output, end='')