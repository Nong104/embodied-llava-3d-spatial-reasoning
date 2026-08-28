import re

import torch


def truncate_at_sentence_end(text: str) -> str:
    """Cut a generated answer at the first sentence-ending punctuation mark
    (., !, or ?), keeping only what comes before it.
    """
    match = re.search(r"[.!?]", text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    image_tokens: torch.Tensor,
    boxes: torch.Tensor,
    category_ids: torch.Tensor,
    prompt: str,
    max_new_tokens: int = 10,
    eos_token_id: int | None = None,
    repetition_penalty: float = 1.3,
) -> str:
    """Greedy-decode a free-form text answer from an EmbodiedLLaVA model.

    repetition_penalty discourages the model from selecting a token it has
    already generated earlier in this same answer, by dividing that token's
    logit (or multiplying, if negative) before the argmax selection. This
    only affects how fluent a single answer reads; it has no bearing on
    whether the answer is factually correct, which depends entirely on what
    the model has learned during training.
    """
    model.eval()
    device = image_tokens.device

    spatial_tokens = model.encode_spatial_tokens(boxes=boxes, category_ids=category_ids)
    fused_tokens = model.fuse_multimodal_tokens(
        image_tokens=image_tokens,
        spatial_tokens=spatial_tokens,
    )

    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(device)
    attention_mask = encoded_prompt["attention_mask"].to(device)

    text_embeddings = model.llava_model.get_input_embeddings()(input_ids)
    fused_tokens = fused_tokens.to(device=text_embeddings.device, dtype=text_embeddings.dtype)

    inputs_embeds = torch.cat([fused_tokens, text_embeddings], dim=1)

    batch_size = input_ids.size(0)
    num_fused_tokens = fused_tokens.size(1)
    fused_attention_mask = torch.ones(
        batch_size, num_fused_tokens, device=device, dtype=attention_mask.dtype
    )
    combined_attention_mask = torch.cat([fused_attention_mask, attention_mask], dim=1)

    generated_ids = []
    all_generated_token_ids = set()

    for _ in range(max_new_tokens):
        outputs = model.llava_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
        )
        logits = outputs.logits[:, -1, :].clone()

        for token_id in all_generated_token_ids:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        next_token_id = logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id)
        all_generated_token_ids.add(next_token_id.item())

        if eos_token_id is not None and bool((next_token_id == eos_token_id).all()):
            break

        next_embed = model.llava_model.get_input_embeddings()(next_token_id)
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
        combined_attention_mask = torch.cat(
            [combined_attention_mask, torch.ones(batch_size, 1, device=device, dtype=combined_attention_mask.dtype)],
            dim=1,
        )

    generated_ids = torch.cat(generated_ids, dim=1)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = [truncate_at_sentence_end(t) for t in generated_text]
    return generated_text[0] if batch_size == 1 else generated_text