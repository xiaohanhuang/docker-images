"""Response generator backends for preference data generation."""


def _get_generator(model_spec: str):
    """Create a generator function for the specified model."""

    def generate_with_transformers(model_id: str):
        """Generator using HuggingFace transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        def generate(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()
            return response

        return generate

    def generate_with_vllm(endpoint_url: str, model_name: str):
        """Generator using vLLM OpenAI-compatible endpoint."""
        from openai import OpenAI

        client = OpenAI(base_url=endpoint_url, api_key="not-required")

        def generate(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            return response.choices[0].message.content

        return generate

    # Parse model specification
    if model_spec.startswith("vllm://"):
        # Format: vllm://model-name@http://endpoint:port
        spec = model_spec.removeprefix("vllm://")
        if "@" not in spec:
            raise ValueError(
                f"Invalid vLLM spec '{model_spec}'. Expected format: "
                "'vllm://model-name@http://endpoint:port'"
            )
        model_name, endpoint = spec.split("@", 1)
        return generate_with_vllm(endpoint, model_name)

    elif model_spec.startswith("hf://"):
        # Format: hf://model-id
        model_id = model_spec.removeprefix("hf://")
        return generate_with_transformers(model_id)

    else:
        # Default: treat as HuggingFace model ID
        return generate_with_transformers(model_spec)
