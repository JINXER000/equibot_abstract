
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from easydict import EasyDict
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_skill_embs(cfg, descriptions, cache_dir):
    """
    Bert embeddings for task embeddings. Borrow from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/lifelong/utils.py#L152.
    """
    if cfg.task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=cache_dir
        )
        skill_embeddings = []
        for description in descriptions:
            tokens = tz(
                text=description,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length=cfg.data.max_word_len,  # maximum length of a sentence
                padding="max_length",
                return_attention_mask=True,  # Generate the attention mask
                return_tensors="pt",  # ask the function to return PyTorch tensors
            )
            masks = tokens["attention_mask"]
            input_ids = tokens["input_ids"]
            task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
                "pooler_output"
            ].detach().cpu().numpy()
            skill_embeddings.append(task_embs)
    else:
        raise ValueError("Unsupported task embedding format")
    return skill_embeddings


def get_skill_bert_embs(skill_names, cache_dir="./data/bert"):
    """
    Generate BERT embeddings for skill names.
    
    Args:
        skill_names: List of skill names to generate embeddings for
        cache_dir: Directory to cache BERT model and embeddings
    
    Returns:
        Dictionary mapping skill names to their BERT embeddings
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "skill_emb_bert.npy")
    
    # Check if cached embeddings exist
    if os.path.exists(cache_file):
        skill_name_to_emb = np.load(cache_file, allow_pickle=True).item()
        # Check if all skill names are in the cache
        if all(skill_name in skill_name_to_emb for skill_name in skill_names):
            return skill_name_to_emb
    
    # Generate embeddings for skill names
    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })
    
    # Generate embeddings using the existing get_skill_embs function
    skill_embs = get_skill_embs(cfg, skill_names, cache_dir)
    
    # Create mapping from skill names to embeddings
    skill_name_to_emb = {skill_names[i]: skill_embs[i] for i in range(len(skill_names))}
    
    # Cache the embeddings
    np.save(cache_file, skill_name_to_emb)
    
    return skill_name_to_emb


class MLPEncoder(nn.Module):
    """
    Encode task embedding

    h = f(e), where
        e: pretrained task embedding from large model
        h: latent embedding (B, H)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        assert num_layers >= 1, "[error] num_layers < 1"
        sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.projection = nn.Sequential(*layers)

    def forward(self, task_emb):
        """
        data:
            task_emb: (B, E)
        """
        h = self.projection(task_emb)  # (B, H)
        return h