import json
import os
import re
import warnings
from pathlib import Path
import torch
from transformers.utils import logging as hf_logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Suppress repetitive C++/CUDA kernel fallback notices and PEFT adapter key warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore', category=UserWarning, module='peft')

class QwenResponse:
    def __init__(self, model_name='Qwen/Qwen3.5-0.8B'):
        # Use CUDA GPU by default if available, or respect HAFIBOT_QWEN_DEVICE
        requested_device = os.getenv('HAFIBOT_QWEN_DEVICE', 'auto').strip().lower()
        if requested_device in ('auto', 'cuda', ''):
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = False
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        base_dir = Path(__file__).resolve().parent.parent
        project_root = base_dir.parent

        # Prioritize QwenModel2, fallback to QwenModel
        model_path_2 = base_dir / 'models' / 'QwenModel2'
        model_path_1 = base_dir / 'models' / 'QwenModel'
        self.model_path = model_path_2 if model_path_2.exists() else model_path_1

        # Locate snapshot if cached locally (check multiple fallback locations)
        candidate_bases = [
            project_root / '.hf_cache' / 'hub' / 'models--Qwen--Qwen3.5-0.8B' / 'snapshots',
            Path('D:/Tugas Akhir/Fall Detection 1D CNN-LSTM/.hf_cache/hub/models--Qwen--Qwen3.5-0.8B/snapshots'),
            Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--Qwen--Qwen3.5-0.8B' / 'snapshots',
        ]
        if os.getenv('HF_HOME'):
            candidate_bases.insert(0, Path(os.environ['HF_HOME']) / 'hub' / 'models--Qwen--Qwen3.5-0.8B' / 'snapshots')

        load_path = model_name
        for snap_base in candidate_bases:
            if snap_base.exists():
                snaps = list(snap_base.iterdir())
                for s in snaps:
                    # Verify actual weight files exist in snapshot
                    if (s / 'model.safetensors-00001-of-00001.safetensors').exists() or (s / 'model.safetensors').exists() or (s / 'pytorch_model.bin').exists():
                        load_path = str(s)
                        break
                if load_path != model_name:
                    break

        self.tokenizer = AutoTokenizer.from_pretrained(load_path if load_path != model_name else str(self.model_path))

        try:
            from transformers import Qwen3_5ForConditionalGeneration
            base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                load_path,
                dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else 'cpu'
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                load_path,
                dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else 'cpu'
            )

        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        self.model.eval()
        print(f'QwenResponse Loaded from {self.model_path.name} on {self.device}')

    def message_builder(self, text):
        ctx = text.get('context', '') if isinstance(text, dict) else ''
        inp = text.get('input', '') if isinstance(text, dict) else str(text)
        return [
            {
                'role': 'system',
                'content': (
                    'Anda adalah Hafibot, robot asisten pendamping lansia yang ramah dan hangat. '
                    'Jawablah secara singkat, jelas, dan santun dalam bahasa Indonesia (maksimal 1-2 kalimat untuk percakapan suara). '
                    'Hindari menggunakan format daftar atau poin-poin panjang.\n'
                    'Informasi internal sistem (jangan disebut kecuali diminta):\n' + str(ctx)
                )
            },
            {
                'role': 'user',
                'content': inp
            }
        ]

    def format_nontest(self, text):
        messages = self.message_builder(text)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return self.tokenizer(prompt, return_tensors='pt')

    def generate_response(self, text):
        sample = self.format_nontest(text)
        sample = {key: value.to(self.model.device) for key, value in sample.items()}
        with torch.inference_mode():
            outputs = self.model.generate(
                **sample,
                max_new_tokens=48,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][sample['input_ids'].shape[1]:], skip_special_tokens=True)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'[\U00010000-\U0010ffff]', '', response).strip()
        return response

if __name__ == '__main__':
    qwen = QwenResponse()
    print(qwen.generate_response({'input': 'Halo robot!', 'context': ''}))
