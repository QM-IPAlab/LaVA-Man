"""
python add_reverse_language_data.py --hdf5_path /data/home/acw694/CLIPort_new_loss/scratch/data_hdf5/bridge_crossview_goal_3imgs_mask_val.hdf5 --dry_run --max_preview 50
"""


import h5py
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

def add_reverse_language(hdf5_path, model, dry_run=False, max_preview=5):
    with h5py.File(hdf5_path, 'a') as f:
        if 'reverse_language' in f and not dry_run:
            print("reverse_language dataset already exists, skipping.")
            return
        
        language_data = f['language'][:]
        reverse_language_data = []
        print(f"Generating reversed instructions for {len(language_data)} entries...")

        prompts = []
        batch_size = 256
        for idx, lang in enumerate(language_data):
            text = lang.decode('utf-8')
            prompt = (
                "You are an instruction-reversing machine that has operated flawlessly for hundreds of years.\n"
                "Your only task is to reverse the direction of object manipulation instructions with perfect precision.\n"
                "You must keep the action and the object unchanged, but swap the start and end locations.\n"
                "Example:\n"
                "Input: move the blue cube from the left side of the table to the right side of the shelf.\n"
                "Output: move the blue cube from the right side of the shelf to the left side of the table.\n\n"
                "Example:\n"
                "Input: open the drawer.\n"
                "Output: close the drawer.\n\n"
                "Next is my instruciton. Just give me the output only\n"
                f"Input: {text}\nOutput:")
            prompts.append([{"role": "user", "content": f"{prompt}"}])
            
            if dry_run and idx >= max_preview:
                break
            
        
        for i in range(0, len(prompts), batch_size):
            num_samples = (i+1) * batch_size
            print(f" Generating reversed instructions in batches of {num_samples}...")
        
            batch_prompts = prompts[i:i+batch_size]      
            try:           
                batch_outputs = model(batch_prompts, batch_size=batch_size)
            except Exception as e:
                print(f"Error generating reversed instructions: {e}")
                batch_outputs = [""] * len(batch_prompts)

            for out in batch_outputs: 
                reversed_text = out[0]['generated_text'][1]['content']
                try:
                    reverse_language_data.append(reversed_text.encode('ascii'))
                except Exception as e:
                    print(f"Error encoding reversed text: {e}, using utf-8")
                    print(reversed_text)
                    reverse_language_data.append(reversed_text.encode('utf-8', errors='replace'))

            if dry_run and i >= max_preview:
                break

        reverse_language_data = np.array(reverse_language_data, dtype=h5py.special_dtype(vlen=bytes))
        
        if dry_run:
            print(f"\n Dry run mode enabled. Previewing first {max_preview} examples:")
            for i in range(min(max_preview, len(reverse_language_data))):
                print(f"Original: {language_data[i].decode('utf-8')}")
                print(f"Reversed: {reverse_language_data[i].decode('utf-8')}\n")
            print(f"Dry run completed")
        else:
            append_or_create_dataset(f, 'reverse_language', data=reverse_language_data, dtype=h5py.special_dtype(vlen=bytes))
            print(f"Successfully added 'reverse_language' with {len(reverse_language_data)} entries.")


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


# Use a pipeline as a high-level helper
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--dry_run', action='store_true', help='Preview the first few examples without writing to the file')
    parser.add_argument('--max_preview', type=int, default=5, help='Number of examples to preview in dry run mode')
    args = parser.parse_args()
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct",max_new_tokens=512, tokenizer=tokenizer, device='cuda')
    add_reverse_language(args.hdf5_path, pipe, args.dry_run, args.max_preview)