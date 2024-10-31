import torch

def load_and_modify_checkpoint(checkpoint_path, keys_to_remove, new_checkpoint_path):
    """
    加载一个checkpoint 删除指定的模块 并保存到新的checkpoint文件中。
    
    Parameters:
    - checkpoint_path (str): 要加载的checkpoint文件路径。
    - keys_to_remove (list of str): 需要删除的模块名称或其前缀列表。例如 ["encoder", "decoder"]。
    - new_checkpoint_path (str): 新的checkpoint文件路径。
    """
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # 删除指定模块的参数
    modified_checkpoint = {k: v for k, v in state_dict.items() if not any(k.startswith(key) for key in keys_to_remove)}

    # 重新保存新的checkpoint
    torch.save(modified_checkpoint, new_checkpoint_path)
    print(f"Modified checkpoint saved to {new_checkpoint_path}")


def print_state_dict_info(checkpoint_path):
    """
    加载 checkpoint 并打印所有 state_dict 中参数的名称和形状。
    
    Parameters:
    - checkpoint_path (str): 要加载的 checkpoint 文件路径。
    """
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # 判断 checkpoint 是否是直接包含 state_dict 的字典
    state_dict = checkpoint if "model" not in checkpoint else checkpoint["model"]

    print("Parameters in State Dict:")
    print("-" * 40)
    
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")


# 使用示例
import pdb; pdb.set_trace()
checkpoint_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra2/checkpoint-160.pth'
new_checkpoint_path = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/output_mae_robot_lang_big_extra2/encoder_only_ck-160.pth'
keys_to_remove = ["decoder"]  # 只保留encoder部分
print_state_dict_info(checkpoint_path)
load_and_modify_checkpoint(checkpoint_path, keys_to_remove, new_checkpoint_path)
