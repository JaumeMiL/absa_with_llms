
import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.common import load_prompt, load_dataset, prepare_messages, encode, generate, extract_json

# renamed to inference.py :)



def generate_text(model, tokenizer, messages, temperature, top_p, max_new_tokens):

    raise NotImplementedError("Generate text not implemented")

"""
Com a referencia, antic feia


def encode(tokenizer, messages) :
    # convert the sequence of messages to an actual prompt in the format
    # expected by the model.
    # Tokenize and encode the prompt.
    input_ids = tokenizer.apply_chat_template(messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
    return input_ids


def generate(model, tokenizer, input_ids):
    # generate likely continuation (assistant answer)
    with torch.no_grad():
        gen_tokens = model.generate(input_ids,
                                    max_new_tokens=256,
                                    pad_token_id=tokenizer.eos_token_id
                                   )
    promptlen = len(input_ids[0])
    # decode obtained tokens back into text
    gen_text = tokenizer.decode(gen_tokens[0][promptlen:], skip_special_tokens=True)
    return gen_text

"""


# NOSE SI POSAR-LO AQUI O EN UN COMMON / BUILDERS o algo aixi pero bno
def load_model_and_tokenizer(model_name, quantization, adapter, device, model_path):
    # Bno aqui hauria de fer lo de la funcio de baix de load_model xd

    # Paaassa que segurament, hauria d'incloure també opcions per
    # carregar adapter si te (de finetune-inference), utilitzar quantization
    # si no hi ha chat template definir-lo
    raise NotImplementedError("Load model and tokenizer not implemented")




# ------------ load model and tokenizer -----------------
def load_model():
    t0 = time.time()

    MODEL_PATH = f"/scratch/nas/1/PDI/mml0/models/llama32B3/"
    # or alternatively
    # f"/scratch/nas/1/PDI/mml0/models/qwen25B3/"

    # load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 dtype=torch.bfloat16,
                                                 device_map="auto")
    # set inference mode                                             
    model.eval()

    # load tokenizer      
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loading took {time.time()-t0:.1f} seconds")
    return model, tokenizer


############## main ###################


print(f"========= ZERO SHOT ===  ")

# load model and tokenizer
model, tokenizer = load_model()

# get system and user prompts
prompts = load_prompt()

# load data to process
examples = load_dataset("devel")

# annotate each example in devel data
t0 = time.time()
for i,ex in enumerate(examples):
    print(f"Processing example {i}", flush=True)
    # prepare sequence of messages for this example
    messages = prepare_messages(prompts, ex)    
    # create example prompt, tokenize, and encode it into tokens
    input_ids = encode(tokenizer, messages)
    # call model to generate response            
    gen_text = generate(model, tokenizer, input_ids)
    # extract json from response
    examples[i]["prediction"] = extract_json(gen_text)

print("Done")
print(f"Processed {len(examples)} examples in {time.time()-t0:.1f} seconds. ({(time.time()-t0)/len(examples):.2f} sec/example)")

# save output
outfname = f"ZS.llama32B3.out.json"
with open(outfname, "w") as of :
    json.dump(examples, of, indent=3, ensure_ascii=False)

# clean up gpu
del model
torch.cuda.empty_cache() 

