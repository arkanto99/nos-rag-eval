from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
class Selene:
    def __init__(self, cache_dir, device='cpu'):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
            device_map="auto",
            cache_dir=cache_dir,
            #quantization_config=quantization_config, # remove to load FP16 model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "AtlaAI/Selene-1-Mini-Llama-3.1-8B",
            cache_dir=cache_dir,
        )
        self.device = device

    def evaluate(self, prompt, temperature=0.01, max_new_tokens=512):
        try:
            # Format the prompt into messages
            messages = [{"role": "user", "content": prompt}]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Get model's max length from tokenizer
            max_length = self.tokenizer.model_max_length

            # Prepare model inputs
            model_inputs = self.tokenizer([text], 
                            truncation=True, 
                            max_length=max_length,
                            return_tensors="pt").to(self.device)

            # Apply attention mask
            attention_mask = model_inputs.attention_mask

            # Generate response
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Extract the newly generated tokens
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # Decode the response
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if response:
                return response
            else:
                return "No response generated"

        except Exception as e:
            print(f"Error in evaluate function: {e}")
            return None

    def parse_atla_response(self, response):
        """
        Parse ATLA model response to extract reasoning and score.

        Args:
            response (str): Raw response from ATLA model

        Returns:
            tuple: (critique, score) where critique is a string and score is an integer
        """
        try:
            # Split into lines and clean up
            lines = [line.strip() for line in response.split('\n') if line.strip()]

            # Extract critique (everything between **Reasoning:** and **Result:**)
            critique = None
            score = None

            for i, line in enumerate(lines):
                if line.startswith("**Reasoning:**"):
                    critique = lines[i].replace("**Reasoning:**", "").strip()
                elif line.startswith("**Result:**"):
                    score = lines[i].replace("**Result:**", "").strip()

            # Remove style tag if present
            if critique and "<userStyle>" in critique:
                critique = critique.split("<userStyle>")[0].strip()

            return critique, score

        except Exception as e:
            print(f"Error parsing ATLA response: {e}")
            return None, None