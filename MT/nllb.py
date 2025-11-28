from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import yaml

codes = {
    'en': 'eng_Latn',
    'de': 'deu_Latn',
}

def parse_args():
    parser = ArgumentParser(description='Translate the source file using a pretrained model.')
    parser.add_argument('input_file', type=str, help='Path to the source file')
    parser.add_argument('--model', type=str, default="facebook/nllb-200-3.3B", help='Path to the pretrained model')
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang=codes[args.src_lang], tgt_lang=codes[args.tgt_lang])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    identifiers = list(data.keys())
    source_sentences = ['\n'.join(data[id]['text']) for id in identifiers]
    
    translations = []
    for i in tqdm(range(0, len(source_sentences), args.batch_size)):
        batch = source_sentences[i:i + args.batch_size]
        inputs = tokenizer(batch, 
                           return_tensors="pt", 
                           padding=True, 
                           truncation=True, 
                           max_length=512).to(device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(codes[args.tgt_lang]),
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