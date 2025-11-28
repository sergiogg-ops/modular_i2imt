from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import yaml

language = {
    'en': 'English',
    'de': 'German',
}

def parse_args():
    parser = ArgumentParser(description='Translate the source file using a pretrained Gemma model.')
    parser.add_argument('input_file', type=str, help='Path to the source file')
    parser.add_argument('--model', type=str, default="google/gemma-3-4b-it", help='Path to the pretrained model')
    parser.add_argument('--output', type=str, default='output.yaml', help='Path to the output file')
    parser.add_argument('-src','--src_lang', type=str, required=True, help='Source language code')
    parser.add_argument('-tgt','--tgt_lang', type=str, required=True, help='Target language code')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for translation')
    args = parser.parse_args()
    return args

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def main():
    args = parse_args()
    data = read_file(args.input_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              token="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",)
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 token="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",
                                                 torch_dtype=torch.bfloat16).to(device).eval()

    identifiers = list(data.keys())
    
    prompts = []
    for id in identifiers:
        text = '\n'.join(data[id]['text'])
        messages = [
            {"role": "user", "content": f"Translate the following {language[args.src_lang]} text \
             to {language[args.tgt_lang]} without further comments:\n{text}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    translations = []
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1,
                do_sample=False
            )
        
        input_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        translations.extend(decoded)

    for id, translation in zip(identifiers, translations):
        data[id]['text'] = [t.strip() for t in translation.split('\n') if t.strip()]

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

if __name__ == '__main__':
    main()
