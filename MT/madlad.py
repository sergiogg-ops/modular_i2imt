from transformers import TranslationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
import yaml

def parse_args():
    parser = ArgumentParser(description='Translate the source file using a pretrained model.')
    parser.add_argument('input_file', type=str, help='Path to the source file')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1", help='Path to the pretrained model')
    parser.add_argument('--output', type=str, default='output.yaml', help='Path to the output file')
    parser.add_argument('-src','--src_lang', type=str, required=True, help='Source language code')
    parser.add_argument('-tgt','--tgt_lang', type=str, required=True, help='Target language code')
    args = parser.parse_args()
    return args

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def main():
    args = parse_args()
    data = read_file(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model, 
                                              src_lang=args.src_lang, 
                                              tgt_lang=args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()
    translation_pipeline = TranslationPipeline(model=model, 
                                               tokenizer=tokenizer, 
                                               batch_size=args.batch_size, 
                                               num_workers=2, 
                                               device='cuda')

    identifiers = list(data.keys())
    source_sentences = [f"<2{args.tgt_lang}> {'\n'.join(data[id]['text'])}" for id in identifiers]
    predictions = translation_pipeline(source_sentences, 
                                       max_length=512, 
                                       num_beams=4, 
                                       early_stopping=True, 
                                       src_lang=args.src_lang, 
                                       tgt_lang=args.tgt_lang)
    translations = [pred['translation_text'] for pred in predictions]


    for id, translation in zip(identifiers, translations):
        data[id]['translation'] = translation.split('\n')

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

if __name__ == '__main__':
    main()