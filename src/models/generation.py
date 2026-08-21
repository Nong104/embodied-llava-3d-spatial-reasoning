import re

import torch


def truncate_at_sentence_end(text: str) -> str:
    """Cut a generated answer at the first sentence-ending punctuation mark
    (., !, or ?), keeping only what comes before it.

    This is a pragmatic safety net for short training runs: teaching a model
    to emit an EOS token at exactly the right point requires the labels to
    supervise that stopping behaviour and enough training steps for it to be
    learned reliably. Until that is in place (or even alongside it, since no
    stopping behaviour is ever perfectly learned), truncating at the first
    sentence boundary keeps EM meaningful by removing the trailing,
    unrelated tokens a short-trained model tends to keep generating after
    it has already produced the correct answer.
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
) -> str:
    """Greedy-decode a free-form text answer from an EmbodiedLLaVA model.

    forward_qa() manually builds inputs_embeds by concatenating fused
    multimodal tokens with text embeddings, which is not a format the
    backbone's own .generate() understands out of the box. This function
    reimplements the minimal autoregressive decoding loop needed to turn
    that custom forward pass into free-form text: predict one token, feed
    it back in, repeat.

    Used for computing real EM / BLEU-4 / CIDEr scores (Section 3.8.1),
    which require an actual generated answer string, not just a
    teacher-forced loss value.
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
    for _ in range(max_new_tokens):
        outputs = model.llava_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
        )
        logits = model.llava_model.lm_head(outputs.last_hidden_state)
        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy: pick the highest-probability token
        generated_ids.append(next_token_id)

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