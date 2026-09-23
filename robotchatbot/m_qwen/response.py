import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class QwenResponse:
    def __init__(self, model_name='Qwen/Qwen3.5-0.8B'):
        # Reserve CUDA for the safety-critical fall/following vision pipeline.
        requested_device = os.getenv('HAFIBOT_QWEN_DEVICE', 'cpu').strip().lower()
        use_cuda = requested_device == 'cuda' and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        base_dir = Path(__file__).resolve().parent.parent
        project_root = base_dir.parent

        # Prioritize QwenModel2, fallback to QwenModel
        model_path_2 = base_dir / 'models' / 'QwenModel2'
        model_path_1 = base_dir / 'models' / 'QwenModel'
        self.model_path = model_path_2 if model_path_2.exists() else model_path_1

        # Locate snapshot if cached locally
        snap_base = project_root / '.hf_cache' / 'hub' / 'models--Qwen--Qwen3.5-0.8B' / 'snapshots'
        load_path = model_name
        if snap_base.exists():
            snaps = list(snap_base.iterdir())
            if snaps:
                load_path = str(snaps[0])

        self.tokenizer = AutoTokenizer.from_pretrained(load_path if load_path != model_name else str(self.model_path))

        try:
            from transformers import Qwen3_5ForConditionalGeneration
            base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                load_path,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else 'cpu'
            )
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                load_path,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
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
                'content': 'Anda adalah robot pendamping lansia.\nInformasi internal dari sistem (jangan disebut kecuali diminta):\n' + str(ctx)
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
        with torch.no_grad():
            outputs = self.model.generate(**sample, max_new_tokens=128)
        response = self.tokenizer.decode(outputs[0][sample['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

if __name__ == '__main__':
    qwen = QwenResponse()
    print(qwen.generate_response({'input': 'Halo robot!', 'context': ''}))
