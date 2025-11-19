from vllm import LLM
from vllm.sampling_params import SamplingParams, BeamSearchParams
from argparse import ArgumentParser
import yaml

def parse_args():
    parser = ArgumentParser(description='Translate the source file using a pretrained model.')
    parser.add_argument('input_file', type=str, required=True, help='Path to the source file')
    parser.add_argument('--model', type=str, default="ByteDance-Seed/Seed-X-PPO-7B", help='Path to the pretrained model')
    parser.add_argument('--output', type=str, default='output.yaml', help='Path to the output file')
    parser.add_argument('-src','--src_lang', type=str, required=True, help='Source language code')
    parser.add_argument('-tgt','--tgt_lang', type=str, required=True, help='Target language code')
    args = parser.parse_args()
    return args

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def forward(model, source, prompt):
    messages = [
        prompt.format(text=seg) for seg in source 
    ]
    # Greedy decoding
    decoding_params = SamplingParams(temperature=0,
                                    max_tokens=512,
                                    skip_special_tokens=True)

    results = model.generate(messages, decoding_params)
    predictions = []
    for res in results:
        res = res.outputs[0].text.strip()
        res = res.split('[COT]')[0].strip()  # Remove CoT explanation if present
        predictions.append(res)
    return predictions

def main():
    args = parse_args()
    data = read_file(args.input_file)
    model = LLM(model=args.model,
            max_num_seqs=512,
            tensor_parallel_size=1,
            enable_prefix_caching=False, #True, 
            gpu_memory_utilization=0.95,
            dtype='half')
    prompt = "Translate the following English sentence into German and explain it in detail:\n{text} <de>"

    identifiers = list(data.keys())
    source_sentences = ['\n'.join(data[id]['text']) for id in identifiers]
    translations = forward(model, source_sentences, prompt)

    for id, translation in zip(identifiers, translations):
        data[id]['translation'] = translation.split('\n')
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

if __name__ == '__main__':
    main()