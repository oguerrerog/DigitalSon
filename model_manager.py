import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizer import load_tokenizer

class ModelManager:
    """
    Loads a causal LM (by default distilgpt2), generates conditioned on retrieved memories,
    and saves/loads checkpoints. Keeps model on the provided device.

    Added:
    - estimate_next_token_entropy(prompt): calcula entropía sobre la distribución del siguiente token.
    - make_curiosity_question(user_input): genera una pregunta simple para pedir más info.
    """

    def __init__(self, model_name="distilgpt2", device=None, gen_cfg=None):
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.tokenizer = load_tokenizer(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.gen_cfg = gen_cfg or {
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95
        }

    def generate_with_context(self, prompt, context_texts):
        """
        Build a single prompt from context_texts + prompt, tokenize and generate.
        Returns the generated string (without the prompt prefix).
        """
        prompt_parts = []
        for i, c in enumerate(context_texts or []):
            prompt_parts.append(f"[RECUERDO {i+1}] {c}")
        prompt_parts.append("[PREGUNTA] " + prompt)
        full_prompt = "\n".join(prompt_parts)

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=self.gen_cfg.get("max_new_tokens", 128),
                temperature=self.gen_cfg.get("temperature", 0.8),
                top_k=self.gen_cfg.get("top_k", 50),
                top_p=self.gen_cfg.get("top_p", 0.95),
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    def estimate_next_token_entropy(self, prompt, context_texts=None):
        """
        Estima la entropía de la distribución del siguiente token dado el prompt.
        Usado para detectar incertidumbre y activar curiosidad.
        Devuelve entropía (float).
        """
        prompt_parts = []
        for i, c in enumerate(context_texts or []):
            prompt_parts.append(f"[RECUERDO {i+1}] {c}")
        prompt_parts.append("[PREGUNTA] " + prompt)
        full_prompt = "\n".join(prompt_parts)

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)  # outputs.logits shape: (batch, seq_len, vocab)
            logits = outputs.logits  # tensor
            # take logits for the last position
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            # compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        return entropy

    def make_curiosity_question(self, user_input):
        """
        Build a short curiosity question based on the user's input.
        This is simple/template-based: it asks for clarification or examples.
        """
        # If the input is long, ask for clarification of a specific fragment; keep it simple:
        snippet = user_input.strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        # Templates
        templates = [
            f"No estoy seguro de entender bien: ¿puedes explicarme qué significa '{snippet}'?",
            f"Me interesa aprender más sobre '{snippet}'. ¿Puedes contarme un poco más o dar un ejemplo?",
            f"¿Por favor me enseñas más acerca de '{snippet}'? No lo conozco bien."
        ]
        # choose simplest template deterministically
        # (we choose based on hash to be consistent)
        idx = abs(hash(snippet)) % len(templates)
        return templates[idx]

    def save_checkpoint(self, path_dir, metadata=None):
        os.makedirs(path_dir, exist_ok=True)
        # Save model and tokenizer in HF format (safe for reload).
        self.model.save_pretrained(path_dir)
        self.tokenizer.save_pretrained(path_dir)
        if metadata is not None:
            with open(os.path.join(path_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, path_dir, device=None):
        device = device or self.device
        self.model = AutoModelForCausalLM.from_pretrained(path_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(path_dir)
        self.model.to(device)
        self.device = device
