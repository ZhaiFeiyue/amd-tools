# vLLM Supported Models

Source: `vllm/model_executor/models/registry.py`

Full list: https://docs.vllm.ai/en/latest/models/supported_models.html

## Text Generation (Decoder-only)

| Model Family | Architecture Class | HuggingFace Example | Type |
|---|---|---|---|
| Llama / Llama-2 / Llama-3 | `LlamaForCausalLM` | `meta-llama/Llama-3-8B-Instruct` | Dense |
| Llama-4 | `Llama4ForCausalLM` | `meta-llama/Llama-4-Scout-17B` | MoE |
| Qwen | `QWenLMHeadModel` | `Qwen/Qwen-7B-Chat` | Dense |
| Qwen2 | `Qwen2ForCausalLM` | `Qwen/Qwen2-7B-Instruct` | Dense |
| Qwen2-MoE | `Qwen2MoeForCausalLM` | `Qwen/Qwen2-57B-A14B-Instruct` | MoE |
| Qwen3 | `Qwen3ForCausalLM` | `Qwen/Qwen3-8B` | Dense |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | `Qwen/Qwen3-30B-A3B` | MoE |
| DeepSeek | `DeepseekForCausalLM` | `deepseek-ai/deepseek-llm-7b` | Dense |
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | `deepseek-ai/DeepSeek-V2` | MoE+MLA |
| DeepSeek-V3 / V3.2 | `DeepseekV3ForCausalLM` | `deepseek-ai/DeepSeek-V3` | MoE+MLA |
| **Kimi-K2 (Linear)** | `KimiLinearForCausalLM` | `moonshotai/Kimi-K2` | MoE+MLA |
| Mixtral | `MixtralForCausalLM` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE |
| Mistral | `MistralForCausalLM` | `mistralai/Mistral-7B-Instruct-v0.3` | Dense |
| Mistral Large 3 | `MistralLarge3ForCausalLM` | `mistralai/Mistral-Large-Instruct-2501` | Dense |
| Gemma | `GemmaForCausalLM` | `google/gemma-7b-it` | Dense |
| Gemma2 | `Gemma2ForCausalLM` | `google/gemma-2-27b-it` | Dense |
| Gemma3 | `Gemma3ForCausalLM` | `google/gemma-3-27b-it` | Dense |
| Gemma3n | `Gemma3nForCausalLM` | `google/gemma-3n-E4B-it` | Dense |
| Phi-3 | `Phi3ForCausalLM` | `microsoft/Phi-3-mini-4k-instruct` | Dense |
| Phi-MoE | `PhiMoEForCausalLM` | `microsoft/Phi-3.5-MoE-instruct` | MoE |
| ChatGLM / GLM-4 | `ChatGLMModel` | `THUDM/chatglm3-6b` | Dense |
| GLM-4 | `Glm4ForCausalLM` | `THUDM/glm-4-9b-chat` | Dense |
| GLM-4 MoE | `Glm4MoeForCausalLM` | `THUDM/GLM-4-MoE` | MoE |
| InternLM2 | `InternLM2ForCausalLM` | `internlm/internlm2-7b` | Dense |
| Falcon | `FalconForCausalLM` | `tiiuae/falcon-40b` | Dense |
| Falcon-H1 | `FalconH1ForCausalLM` | `tiiuae/Falcon-H1-34B` | Hybrid |
| Baichuan / Baichuan2 | `BaichuanForCausalLM` | `baichuan-inc/Baichuan2-13B-Chat` | Dense |
| Bloom | `BloomForCausalLM` | `bigscience/bloom` | Dense |
| OPT | `OPTForCausalLM` | `facebook/opt-6.7b` | Dense |
| GPT-2 | `GPT2LMHeadModel` | `openai-community/gpt2` | Dense |
| GPT-J | `GPTJForCausalLM` | `EleutherAI/gpt-j-6b` | Dense |
| GPT-NeoX | `GPTNeoXForCausalLM` | `EleutherAI/gpt-neox-20b` | Dense |
| StarCoder2 | `Starcoder2ForCausalLM` | `bigcode/starcoder2-15b` | Dense |
| Cohere Command-R | `CohereForCausalLM` | `CohereForAI/c4ai-command-r-plus` | Dense |
| OLMo | `OlmoForCausalLM` | `allenai/OLMo-7B` | Dense |
| OLMo2 / OLMo3 | `Olmo2ForCausalLM` | `allenai/OLMo2-7B` | Dense |
| OLMoE | `OlmoeForCausalLM` | `allenai/OLMoE-1B-7B-0924` | MoE |
| Mamba | `MambaForCausalLM` | `state-spaces/mamba-2.8b` | SSM |
| Mamba2 | `Mamba2ForCausalLM` | `state-spaces/mamba2-2.7b` | SSM |
| Jamba | `JambaForCausalLM` | `ai21labs/Jamba-v0.1` | Hybrid |
| DBRX | `DbrxForCausalLM` | `databricks/dbrx-instruct` | MoE |
| Granite | `GraniteForCausalLM` | `ibm-granite/granite-3b-code-instruct` | Dense |
| Granite MoE | `GraniteMoeForCausalLM` | `ibm-granite/granite-moe` | MoE |
| MiniCPM | `MiniCPMForCausalLM` | `openbmb/MiniCPM-2B-sft-bf16` | Dense |
| MiniCPM3 | `MiniCPM3ForCausalLM` | `openbmb/MiniCPM3-4B` | Dense |
| MiniMax-Text01 | `MiniMaxText01ForCausalLM` | `MiniMaxAI/MiniMax-Text-01` | MoE |
| MiniMax-M2 | `MiniMaxM2ForCausalLM` | `MiniMaxAI/MiniMax-M2` | MoE |
| Arctic | `ArcticForCausalLM` | `Snowflake/snowflake-arctic-instruct` | MoE |
| Grok-1 | `Grok1ModelForCausalLM` | `xai-org/grok-1` | MoE |
| Nemotron | `NemotronForCausalLM` | `nvidia/Nemotron-4-340B-Instruct` | Dense |
| HunYuan MoE | `HunYuanMoEV1ForCausalLM` | `tencent/Hunyuan-A13B-Instruct` | MoE |
| Bailing MoE | `BailingMoeForCausalLM` | `inclusionAI/Ling-lite` | MoE |
| Step-1 | `Step1ForCausalLM` | `stepfun-ai/Step1-8B` | Dense |
| MiMo | `MiMoForCausalLM` | `XiaomiMiMo/MiMo-7B-RL` | Dense |
| Ernie 4.5 | `Ernie4_5ForCausalLM` | `baidu/ERNIE-4.5-0.3B-Instruct` | Dense |
| Ernie 4.5 MoE | `Ernie4_5_MoeForCausalLM` | `baidu/ERNIE-4.5-21B-A3B-Instruct` | MoE |
| JAIS | `JAISLMHeadModel` | `inceptionai/jais-13b-chat` | Dense |
| Solar | `SolarForCausalLM` | `upstage/SOLAR-10.7B-Instruct-v1.0` | Dense |
| TeleChat2 | `TeleChat2ForCausalLM` | `Tele-AI/TeleChat2-7B` | Dense |
| Zamba2 | `Zamba2ForCausalLM` | `Zyphra/Zamba2-7B-instruct` | Hybrid |

## Multimodal Models (Vision / Audio / Video)

| Model Family | Architecture Class | Modalities | Type |
|---|---|---|---|
| **Kimi-K2.5** | `KimiK25ForConditionalGeneration` | Text + Image + Video | MoE+MLA |
| **Kimi-VL** | `KimiVLForConditionalGeneration` | Text + Image | MoE |
| **Kimi-Audio** | `KimiAudioForConditionalGeneration` | Text + Audio | MoE |
| Qwen2-VL | `Qwen2VLForConditionalGeneration` | Text + Image + Video | Dense |
| Qwen2.5-VL | `Qwen2_5_VLForConditionalGeneration` | Text + Image + Video | Dense |
| Qwen3-VL | `Qwen3VLForConditionalGeneration` | Text + Image + Video | Dense |
| Qwen3-VL-MoE | `Qwen3VLMoeForConditionalGeneration` | Text + Image + Video | MoE |
| Qwen2-Audio | `Qwen2AudioForConditionalGeneration` | Text + Audio | Dense |
| Qwen2.5-Omni | `Qwen2_5OmniForConditionalGeneration` | Text + Image + Audio + Video | Dense |
| Qwen3-Omni-MoE | `Qwen3OmniMoeForConditionalGeneration` | Text + Image + Audio + Video | MoE |
| Gemma3 (MM) | `Gemma3ForConditionalGeneration` | Text + Image | Dense |
| Llama-4 (MM) | `Llama4ForConditionalGeneration` | Text + Image | MoE |
| InternVL | `InternVLChatModel` | Text + Image + Video | Dense |
| LLaVA | `LlavaForConditionalGeneration` | Text + Image | Dense |
| LLaVA-NeXT | `LlavaNextForConditionalGeneration` | Text + Image | Dense |
| LLaVA-NeXT-Video | `LlavaNextVideoForConditionalGeneration` | Text + Image + Video | Dense |
| LLaVA-OneVision | `LlavaOnevisionForConditionalGeneration` | Text + Image + Video | Dense |
| Phi-3-Vision | `Phi3VForCausalLM` | Text + Image | Dense |
| Phi-4-MM | `Phi4MMForCausalLM` | Text + Image + Audio | Dense |
| Pixtral | `PixtralForConditionalGeneration` | Text + Image | Dense |
| Mistral-3 (MM) | `Mistral3ForConditionalGeneration` | Text + Image | Dense |
| DeepSeek-VL2 | `DeepseekVLV2ForCausalLM` | Text + Image | MoE |
| GLM-4V | `GLM4VForCausalLM` | Text + Image | Dense |
| MiniCPM-V | `MiniCPMV` | Text + Image + Video | Dense |
| MiniCPM-O | `MiniCPMO` | Text + Image + Audio + Video | Dense |
| Molmo | `MolmoForCausalLM` | Text + Image | Dense |
| Molmo2 | `Molmo2ForConditionalGeneration` | Text + Image | Dense |
| PaliGemma | `PaliGemmaForConditionalGeneration` | Text + Image | Dense |
| BLIP-2 | `Blip2ForConditionalGeneration` | Text + Image | Dense |
| Fuyu | `FuyuForCausalLM` | Text + Image | Dense |
| Aria | `AriaForConditionalGeneration` | Text + Image | MoE |
| HunYuan-Vision | `HunYuanVLForConditionalGeneration` | Text + Image | Dense |
| NVLM-D | `NVLM_D` | Text + Image | Dense |
| Ultravox | `UltravoxModel` | Text + Audio | Dense |
| Whisper | `WhisperForConditionalGeneration` | Audio → Text | Encoder-Decoder |

## Embedding Models

| Model Family | Architecture Class | Type |
|---|---|---|
| BERT | `BertEmbeddingModel` | Encoder |
| RoBERTa / XLM-RoBERTa | `RobertaEmbeddingModel` | Encoder |
| GTE (Snowflake) | `SnowflakeGteNewModel` | Encoder |
| NomicBERT | `NomicBertModel` | Encoder |
| ModernBERT | `ModernBertModel` | Encoder |
| E5-Mistral (Llama-based) | `LlamaForCausalLM` | Decoder |
| BGE-M3 | `BgeM3EmbeddingModel` | Encoder |
| CLIP | `CLIPEmbeddingModel` | Multimodal |
| SigLIP | `SiglipEmbeddingModel` | Multimodal |
| GritLM | `GritLM` | Decoder |
| ColPali | `ColPaliModel` | Multimodal |
| ColQwen3 | `ColQwen3Model` | Multimodal |

## Reward / Classification Models

| Task | Models |
|---|---|
| Reward Modeling | InternLM2, Qwen2 |
| Token Classification | BERT, Ernie, ModernBERT |
| Sequence Classification | BERT, GPT-2, Ernie, Jamba, Llama, RoBERTa, ModernBERT |

## Model Architecture Notes

### Dense vs MoE
- **Dense**: All parameters activated per token (e.g., Llama-3 70B → 70B per token)
- **MoE**: Only subset activated per token (e.g., Kimi-K2.5: 1T total, 32B activated)
- MoE models benefit from Expert Parallelism (`--enable-expert-parallel`)

### MLA (Multi-head Latent Attention)
- Used by: DeepSeek-V2/V3, Kimi-K2/K2.5
- Compresses KV cache via low-rank projection, reducing memory
- Special attention backends: TRITON_MLA, FlashInfer MLA

### Hybrid Models (Transformer + SSM)
- Jamba, Falcon-H1, Zamba2, Bamba: Combine attention layers with Mamba/SSM layers
- Require special KV cache handling (hybrid KV cache manager)

## How to Check Model Support

```python
# Check if a model is supported
from vllm import LLM
llm = LLM(model="<model-name>")  # will error if unsupported

# Or check the registry directly
from vllm.model_executor.models import ModelRegistry
print(ModelRegistry.is_text_generation_model("LlamaForCausalLM"))
```

```bash
# List registered models via CLI
python -c "from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS; print('\n'.join(sorted(_TEXT_GENERATION_MODELS.keys())))"
```

## Adding Custom / Unsupported Models

vLLM supports running models via the **Transformers backend** even without native implementation:

```bash
vllm serve <model> --trust-remote-code
```

For native implementation, see the [Model Implementation Guide](https://docs.vllm.ai/en/stable/contributing/model/).
