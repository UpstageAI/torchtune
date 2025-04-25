# System Architecture

```text
                                ┌─────────────────────────────────────────┐
                                │ 3. Builders & High‑Level API            │
                                ├─────────────────────────────────────────┤
                                │ docev_encoder_with_connector(args)      │
                                │   # → calls: docev_vision_encoder(),    │
                                │            docev_ldp_v2_connector()     │
                                │ docev_solar_decoder(args)               │
                                │   # → builds: FusionEmbedding,          │
                                │               TransformerSelfAttention  │
                                │ EarlyFusionModel(decoder, encoder,…)    │
                                │   # → wraps `<image>` 토큰 로직           │
                                │ lora_docev_* variants                   │
                                │   # → PEFT 방식으로 부분 LoRA 적용          │
                                └─────────────────────────────────────────┘
                                          ▲
                                          │
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            2. Model Architecture (Core)                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  Vision Encoder      Connector      EncoderWithConnector           Decoder              │
│  (SiglipViT)         (DocEVLDPv2)   (DocEVEncoderWithConnector)   (TransformerDecoder)  │
│  Conv2d patch embed  MLP → GELU     Vision Encoder → Connector    FusionEmbedding       │
│  TokenPosEmbedding   avg_pool2d     → Packed Sequence             → Decoder layers      │
│  Transformers × N     PEG conv                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │
     ┌────────────────────────────────────┴────────────────────────────┐
     │             1. Data Pipeline                                    │
     │ Raw Sample → UfxToMessages → DocEVTransform → Collate → Inputs  │
     └─────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │
┌────────────────────────────────────────────────────────────────────────┐
│                  4. Utilities & Conversion (Side‑Box)                  │
├────────────────────────────────────────────────────────────────────────┤
│ • docev_hf_to_tune(state_dict) ↔ HuggingFace ↔ torchtune 형식 변환       │
│ • docev_tune_to_hf(state_dict) ↔ torchtune ↔ HuggingFace               │
│ • _utils.py: select_best_resolution(), get_padding(), unpad_image(),...│
└────────────────────────────────────────────────────────────────────────┘
```
## 1. Data Pipeline

**목표:**
- 원시 데이터(텍스트와 이미지 파일)를 모델이 처리 가능한 배치 텐서 형태로 변환.

### 1.1 UfxToMessages
- **구현위치:** : `torchtune.datasets.multimodal._ufx.py`
- **입력:**
  - `sample` 객체: `{ "context": List[dict], "image_files": List[str] }`
- **주요 역할:**
  - UfxToMessages function supports two types of context format:
    - `Context format type 1`
      - Content type is string, If image is present, placeholder for image is prepended to the text
    - `Context format type 2`
      - Content type is list, If image is present, image is appended to the list
    For example:
    ```python
    context_format_type_1 = [
        {
            "role": "user",
            "content": "<image> What is shown in this image?"
        },
        {
            "role": "assistant",
            "content": "There is a red stop sign in the image."
        },
    ]

    context_format_type_2 = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "There is a red stop sign in the image."},
            ],
        },
    ]
    ```
    - Message object will be converted to::
    ```python
        [
            Message(
                role = "user",
                content = [
                    {"type": "image", "content": <PIL.Image.Image>},
                    {"type": "text", "content": "What is shown in this image?"},
                ],
            ),
            Message(
                role = "assistant",
                content = [
                    {"type": "text", "content": "There is a red stop sign in the image."},
                ],
            ),
            ...
        ]
    ```
- **출력:**
  - `{'messages': List[Message]}` 형태의 딕셔너리.

### 1.2 DocEVTransform
- **구현위치:** : `torchtune.models.docev._transform.py`
- **입력:**
  - `{'messages': List[Message]}`
- **주요 역할:**
  - `UfxToMessages` 출력을 입력으로 받아 토큰화 및 이미지 전처리 수행.
  1. **Tokenization**
     - `tokenizer.apply_chat_template()`를 통해 사용된 채팅 포맷 적용.
     - chat_template는 [Jinja Template](https://jinja.palletsprojects.com/en/stable/templates/)를 따름.
  2. **Image Preprocessing**
     - `LlavaNextImageProcessor`로 일링(tiles), 정규화(normalize).
  3. **Handling image tokens**
     - Vision Token 수를 계산하고 이를 토큰화 결과에 반영.
- **출력 키:**
  - `tokens`, `mask`, `encoder_input.images`, `encoder_input.image_sizes`, `encoder_input.sampling_ratio`.

### 1.3 Collate 단계
- **구현위치:** : `torchtune.models.docev._collate.py`
- **padded_collate_sft**
  - batch 내 최대 토큰 수를 기준으로 패딩 처리.
  - `List[{'tokens': List[int], 'labels': List[int]}]` → `(bsz, max_len)` 형태의 LongTensor
  - `padding_idx`, `ignore_idx`, `pad_to_multiple_of` 옵션 지원.
- **padded_collate_tiled_images_and_mask**
  - `pad_direction`: `"right"`(훈련), `"left"`(추론)
  - 내부적으로 `pad_sequence` 또는 `left_pad_sequence` 사용.
  - 이미지: `torch.stack` 및 `torch.concat`으로 `(bsz, n_imgs, tiles, c, h, w)` → `(bsz*n_imgs, tiles*pooled_hw, dim)` 텐서 생성.
  - 이미지 사이즈: `(bsz, n_imgs, 2)` 텐서 생성.
  - Sampling ratio: `(bsz, n_imgs)` 텐서 생성.

---

## 2. Model Architecture (Core)

**목표:**
- 이미지와 텍스트를 효율적으로 처리하여 텍스트를 생성하는 모듈 구성.

### 2.1 Vision Encoder
- **구현위치:** : `torchtune.models.docev._encoder.py`
- **클래스:** `SiglipVisionTransformer`
- **구성 요소:**
  1. **Conv2d patch embedding**: `in_channels → embed_dim`, `kernel_size=patch_size`
  2. **TokenPosEmbedding**: 패치별 위치 임베딩
  3. **TransformerSelfAttentionLayer** × `num_layers`
     - `MultiHeadAttention`: Q/K/V 프로젝션, Rotary/rope 옵션
     - MLP: `hidden_dim` → `embed_dim`
  4. **CLSProjection** (옵션): 최종 CLS 벡터 추출

### 2.2 Connector
- **구현위치:** : `torchtune.models.docev._encoder.py`
- **클래스:** `DocEVLDPv2Connector`
- **단계:**
  1. **MLP projection**: `clip_embed_dim → decoder_embed_dim` + GELU
  2. **Reshape**: `[bsz, imgs, tiles, num_features, dim]` → `[bsz*imgs*tiles, dim, side_len, side_len]`
  3. **avg_pool2d**: `sampling_ratio`만큼 다운샘플링 `[bsz*imgs*tiles, dim, side_len//sampling_ratio, side_len//sampling_ratio]`
  4. **PEG conv**: 그룹별 컨볼루션을 통해 위치 정보 보강
  5. **Flatten & Reshape**: `(bsz*imgs, tiles*pooled_hw, llm_hidden_size)`

### 2.3 DocEVEncoderWithConnector
- **구현위치:** : `torchtune.models.docev._encoder.py`
- **클래스:** `DocEVEncoderWithConnector`
- **역할:** Vision Encoder와 Connector를 결합하여 이미지 피처를 LLM이 처리할 수 있는 시퀀스로 패킹
- **단계:**
  1. **Vision Encoder로부터 [bsz, imgs, tiles, num_features, dim] 형태의 피처 획득**
  2. **Connector (DocEVLDPv2Connector) 호출하여 [bsz*imgs, tiles*pooled_hw, llm_hidden_size] 생성**
  3. **pack_image_features**:
    - base_image_feature와 downsampled feature를 조합
    - newline token 삽입 (이미지 구분자 역할)
    - 전체 이미지 시퀀스를 flatten하여 `[all_feat_len, llm_hidden_size]` 및 각각의 feature_lens 반환

### 2.4 Decoder
- **구현위치:** : `torchtune.modules.transformer.py`
- **클래스:** `TransformerDecoder`
- **구성 요소:**
  - `FusionEmbedding`: 텍스트/이미지 퓨전 토큰 임베딩
  - `TransformerDecoder`:
    - `num_layers` × `TransformerSelfAttentionLayer`
    - `RMSNorm` 노말라이제이션
    - 최종 `Linear` 출력 → 어휘 수

---

## 3. Builders & High‑Level API

### 3.1 Component Builders
- **구현위치:** : `torchtune.models.docev._component_builder.py`
> **Vision Encoder Builder**
```python
docev_vision_encoder(
  tile_size: int,
  patch_size: int,
  num_layers: int,
  embed_dim: int,
  hidden_dim: int,
  num_heads: int,
  activation: Callable,
  cls_output_dim: int,
  attn_bias: bool,
  use_rope: bool,
  out_indices: Optional[List[int]],
  output_cls_projection: bool,
  max_num_tiles: int,
  in_channels: int,
  append_cls_token: bool,
) -> SiglipVisionTransformer
```
- **설명:** 문서 이미지 타일을 패치 단위로 분할하여 임베딩하고, 다수의 Transformer 레이어로 처리하는 ViT 기반 인코더를 생성합니다.
- **주요 파라미터:**
  - `tile_size`: 이미지 타일 크기 설정
  - `patch_size`: conv2d 패치 크기
  - `num_layers`: Transformer 레이어 개수
  - `embed_dim`, `hidden_dim`, `num_heads`: 임베딩 및 MLP 차원과 어텐션 헤드 수
  - `use_rope`: 로터리 위치 인코딩 사용 여부
  - `out_indices`: 중간 레이어 출력 인덱스 지정 (범위 : 0 ~ `num_layers - 1`)
  - `output_cls_projection`: CLS 토큰 출력 프로젝션 여부 (default: False)
  - `max_num_tiles`: 최대 타일 수
  - `in_channels`: 입력 이미지 채널 수
  - `append_cls_token`: CLS 토큰 추가 여부 (default: False)

> **Connector Builder**
```python
docev_ldp_v2_connector(
  clip_embed_dim: int,
  decoder_embed_dim: int,
) -> DocEVLDPv2Connector
```
- **설명:** CLIP 인코더로부터 나온 비전 피처를 LLM 입력 차원으로 투영한 뒤, `sampling_ratio` 기반 다운샘플링과 위치 인코딩 컨볼루션을 적용하여 시퀀스 형태로 변환합니다.
- **주요 파라미터:**
  - `clip_embed_dim`: CLIP 임베딩 차원
  - `decoder_embed_dim`: LLM 디코더 임베딩 차원

> **Encoder with Connector**
```python
docev_encoder_with_connector(
  # Vision encoder args,
  patch_size: int,
  num_heads: int,
  clip_embed_dim: int,
  ...,
  # Connector args,
  decoder_embed_dim: int,
  connector_type: Literal["ldp_v2"],
  ...
) -> DocEVEncoderWithConnector
```
- **설명:** Vision Encoder와 Connector를 결합하여 하나의 엔코더 모듈로 묶습니다. `forward()` 호출 시 이미지 텐서를 받아서 최종 LLM 입력 시퀀스를 반환합니다.
- **출력:** `DocEVEncoderWithConnector` 객체, `forward(images, image_sizes, sampling_ratio)`로 시퀀스 피처와 길이를 출력
- 지원가능한 connector 타입은 현재 `"ldp_v2"` 밖에 없습니다.

> **Decoder Builder**
```python
docev_solar_decoder(
  vocab_size: int,
  fusion_vocab_size: int,
  num_layers: int,
  num_heads: int,
  num_kv_heads: int,
  embed_dim: int,
  max_seq_len: int,
  intermediate_dim: Optional[int],
  attn_dropout: float,
  norm_eps: float,
  rope_base: int,
) -> TransformerDecoder
```
- **설명:** Fusion 토큰 임베딩과 다수의 Transformer Decoder 레이어, 최종 출력 프로젝션을 포함하는 디코더를 생성합니다. 이미지 및 텍스트 임베딩을 함께 처리할 수 있습니다.
- **주요 파라미터:**
  - `fusion_vocab_size`: 이미지 토큰용 퓨전 어휘 크기. 기존 vocab에서 제공하지 않는 token을 추가하여 해당 token embedding만 학습할 수 있습니다. 예를들어 기존 vocab_size가 64000 이고, multi-modal token을 64개 추가하면 최종 vocab_size는 64064가 됩니다. 이후 학습 config에서 `fusion_trainable = True`로 설정하면 해당 token embedding이 학습됩니다.
  - `num_kv_heads`: 키/값 헤드 수 조정(Grouped‑Query Attention 지원)
  - `max_seq_len`, `rope_base`: positional embedding 범위 설정

> **LoRA Component Builders**
```python
lora_docev_ldp_v2_connector(...)
# DocEVLDPv2Connector에 LoRA/DoRA 레이어 추가
lora_docev_encoder_with_connector(...)
# Encoder+Connector에 LoRA 적용
lora_docev_solar_decoder(...)
# TransformerDecoder에 LoRA 적용
```
- **설명:** PEFT(LoRA/DoRA) 기법으로 어텐션 프로젝션과 MLP에 저랭크 적응 계층을 삽입하여 효율적인 파인튜닝을 지원합니다.

### 3.2 Model Builders
- **구현위치:** : `torchtune.models.docev._model_builder.py`
> **Transform Builder**
```python
docev_preview_transform(
  model_name_or_path: str,
  image_token: str,
  stop_tokens: List[str],
  tile_size: int,
  patch_size: int,
  max_num_tiles: int,
  min_num_tiles: int = 1,
  vision_feature_select_strategy: Literal["default","full"],
  sampling_ratio: List[int],
  apply_random_sampling_ratio: bool,
  max_seq_len: Optional[int],
  chat_template: Optional[str]
) -> DocEVTransform
```
- **설명:**
  - HuggingFace 토크나이저와 LlavaNext 이미지 프로세서를 초기화하여, 텍스트-이미지 혼합 입력을 모델에 적합한 형태로 전처리하는 변환기를 생성합니다.
  - 채팅 템플릿, 이미지 토큰, 정지 토큰(stop_tokens) 등을 등록하며, 타일링·정규화·토큰화 과정을 포함합니다.
- **입력:**
  - `model_name_or_path`: 사전학습 모델 경로 혹은 HF 식별자
  - `image_token`: 입력 시 이미지 위치를 표시할 특수 토큰
  - `stop_tokens`: 토큰화 중단 시점을 지정하는 토큰 목록
  - `vision_feature_select_strategy`: 이미지 피처 선택 전략
  - `sampling_ratio`: 타일링 후 다운샘플링 비율
  - `apply_random_sampling_ratio`: 랜덤 다운샘플링 적용 여부
  - `max_seq_len`: 최대 시퀀스 길이
  - `chat_template`: 채팅 템플릿
- **출력:**
  - `DocEVTransform` 객체, `__call__(sample)`로 메시지를 모델 입력(dict) 형태로 변환

> **Early Fusion Model Builder**
```python
docev_preview(
  image_token_id: int,
  decoder_trainable: bool,
  encoder_trainable: bool | Dict[str,bool],
  fusion_trainable: bool
) -> EarlyFusionModel
```
- **설명:**
  - 컴포넌트 빌더에서 생성된 `DocEVEncoderWithConnector`와 `TransformerDecoder`를 결합하여, `<image>` 토큰 기반 멀티모달 인퍼런스/학습을 수행하는 최종 모델 인스턴스를 생성합니다.
  - `decoder_trainable`, `encoder_trainable`, `fusion_trainable` 플래그로 각 파트의 학습 가능 여부를 제어합니다.
- **입력:**
  - `image_token_id`: 토크나이저에서 부여한 이미지 토큰 ID
  - `decoder_trainable`: 디코더 파라미터 학습 여부
  - `encoder_trainable`: 인코더 파라미터 학습 여부 (True/False 또는 세부 dict)
  - `fusion_trainable`: 퓨전(이미지+텍스트 임베딩) 파라미터 학습 여부
- **출력:**
  - `EarlyFusionModel` 객체, `forward(tokens=None, encoder_input=...)`로 이미지와 텍스트를 함께 처리

> **LoRA Model Builder**
```python
lora_docev_preview(
  lora_attn_modules: List[LORA_ATTN_MODULES],
  image_token_id: int,
  decoder_trainable: str,
  encoder_trainable: str,
  fusion_trainable: str,
  apply_lora_to_mlp: bool,
  apply_lora_to_output: bool,
  lora_rank: int,
  lora_alpha: int,
  lora_dropout: float,
  use_dora: bool,
  quantize_base: bool
) -> EarlyFusionModel
```
- **설명:**
  - `docev_preview`에서 생성된 EarlyFusionModel에 PEFT(LoRA/DoRA)를 적용하여, 어텐션 및 MLP 레이어에 경량 적응 계층을 추가한 모델을 반환합니다.
  - `decoder_trainable`, `encoder_trainable`, `fusion_trainable`은 "full", "lora", "frozen" 옵션을 지원하여 세부 제어 가능합니다.
- **출력:**
  - LoRA가 적용된 `EarlyFusionModel` 객체

> **QLoRA Shortcut**
```python
qlora_docev_preview = partial(
  lora_docev_preview,
  quantize_base=True
)
```
- **설명:**
  - `quantize_base=True`로 설정된 QLoRA 구성을 미리 지정한 편의 함수로, 4-bit 양자화를 동반한 LoRA 모델을 손쉽게 생성합니다。


---

## 4. Utilities & Conversion (Side‑Box)

**목표:**
- 외부 포맷 ↔ 내부 포맷 변환 및 부가적인 헬퍼 함수

- **State dict 변환:**
  - `docev_hf_to_tune(state_dict)`: HuggingFace → torchtune 네이밍/포맷
  - `docev_tune_to_hf(state_dict)`: torchtune → HuggingFace
- **_utils.py**:
  - `select_best_resolution()`: 이미지 리사이징 후 최적 그리드 선택
  - `get_padding()`: image 실제 사이즈를 고려하여 padding 크기 계산
  - `unpad_image()`: 실제 image size를 고려하여 connector output에서 padding 제거
  - `get_encoder_output_feature_size()`: 이미지 사이즈에 따른 피처 수 계산
- **_transform.py**:
  - `UfxToMessages`: Message 객체 변환 로직

---
