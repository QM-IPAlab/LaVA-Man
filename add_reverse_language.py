"""
add reverse language for data augmentation on existing hdf5 files
usage:
    python add_reverse_language.py --hdf5_path bridge_crossview_goal.hdf5 --dry_run --max_preview 50
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import h5py
import numpy as np

def append_or_create_dataset(f, name, data, dtype=None):
    if name in f:
        # If dataset already exists, append to it
        dset = f[name]
        dset.resize(dset.shape[0] + len(data), axis=0)
        dset[-len(data):] = data
        n = len(dset)

    else:
        if dtype is None:
            maxshape = (None,) + data[0].shape
            chunks = (1,) + data[0].shape
            f.create_dataset(name, data=data, maxshape=maxshape,
                                chunks=chunks)
        else:
            maxshape = (None,)
            chunks = (1,)  # For variable-length data
            f.create_dataset(name, data=data, maxshape=maxshape,
                                chunks=chunks, dtype=dtype)

        n = len(data)

    return n


def add_reverse_language(hdf5_path, model, dry_run=False, max_preview=5, batch_size=10):
    with h5py.File(hdf5_path, 'a') as f:
        if 'reverse_language' in f:
            print("reverse_language dataset already exists, skipping.")
            return
        
        language_data = f['language'][:]
        reverse_language_data = []
        print(f"Generating reversed instructions for {len(language_data)} entries...")

        prompts = []
        for idx, lang in enumerate(language_data):
            text = lang.decode('utf-8')
            prompt = (
                "You are an instruction-reversing machine that has operated flawlessly for hundreds of years.\n"
                "Your only task is to reverse the given instructions naturally with perfect precision.\n"
                "I will give you an instruction of some actions, and you task is to think about the revert action and generate the reverted instruction accordingly.\n"
                "When you face difficult instructions just try your best.\n"
                "Please only give me the answer in english what could be decoded by ascii.\n"
                "I know you can do it. You are always the best\n"
                "Remenber to keep the object unchanged.\n"
                "Example:\n"
                "Input: move the blue cube from the left side of the table to the right side of the shelf.\n"
                "Output: move the blue cube from the right side of the shelf to the left side of the table.\n\n"
                "Example:\n"
                "Input: open the drawer.\n"
                "Output: close the drawer.\n\n"
                "Example:\n"
                "Input: put pear in bowl.\n"
                "Output: take the pear out of bolw.\n\n"
                "Next is my instruciton. Just give me the output only"
                f"Input: {text}\nOutput:")
            prompts.append([{"role": "user", "content": f"{prompt}"}])
            if dry_run and idx >= max_preview:
                break
            
        print(f"🚀 Generating reversed instructions in batches of {batch_size}...")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]        
            try:           
                batch_outputs = model(batch_prompts, max_new_tokens=512)
            except Exception as e:
                print(f"Error generating reversed instructions: {e}")
                batch_outputs = [""] * len(batch_prompts)
            print(f"Generated {i} instructions")

            for out in batch_outputs: 
                reversed_text = out[0]['generated_text'][1]['content']
                reverse_language_data.append(reversed_text.encode('utf-8'))

        reverse_language_data = np.array(reverse_language_data, dtype=h5py.special_dtype(vlen=bytes))
        
        if dry_run:
            print(f"\n Dry run mode enabled. Previewing first {max_preview} examples:")
            for i in range(min(max_preview, len(reverse_language_data))):
                print(f"Original: {language_data[i].decode('utf-8')}")
                print(f"Reversed: {reverse_language_data[i].decode('utf-8')}\n")
                if i > 100:
                    break
            print(f"Dry run completed")
        else:
            append_or_create_dataset(f, 'reverse_language', data=reverse_language_data, dtype=h5py.special_dtype(vlen=bytes))
            print(f"Successfully added 'reverse_language' with {len(reverse_language_data)} entries.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--dry_run', action='store_true', help='Preview the first few examples without writing to the file')
    parser.add_argument('--max_preview', type=int, help='Number of samples preview in dry run mode')
    parser.add_argument('--batch_size', type=int, help='Number of samples preview in dry run mode')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.padding_side = "left"
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", device='cuda', batch_size = args.batch_size, tokenizer=tokenizer)
    #pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]


    add_reverse_language(args.hdf5_path, pipe, args.dry_run, args.max_preview, args.batch_size)
