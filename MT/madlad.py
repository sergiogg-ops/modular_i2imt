from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import yaml

def parse_args():
    parser = ArgumentParser(description='Translate the source file using a pretrained model.')
    parser.add_argument('input_file', type=str, help='Path to the source file')
    parser.add_argument('--model', type=str, default="google/madlad400-3b-mt", help='Path to the pretrained model')
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    identifiers = list(data.keys())
    source_sentences = ['\n'.join(data[id]['text']) for id in identifiers]
    source_sentences = [f"<2{args.tgt_lang}> {sent}" for sent in source_sentences]
    
    translations = []
    for i in tqdm(range(0, len(source_sentences), args.batch_size)):
        batch = source_sentences[i:i + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Remove leading space token (805) if present, as T5Tokenizer adds it but MADLAD expects <2xx> at start
        if inputs.input_ids.shape[1] > 0 and inputs.input_ids[0, 0] == 805:
            inputs.input_ids = inputs.input_ids[:, 1:]
            inputs.attention_mask = inputs.attention_mask[:, 1:]

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=2,
                early_stopping=True
            )

        
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translations.extend(decoded)


    for id, translation in zip(identifiers, translations):
        data[id]['text'] = translation.split('\n')

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

if __name__ == '__main__':
    main()