# DocEV Preview Training Configuration

이 문서는 TorchTune을 사용해 DocEV Preview 모델을 fine-tuning하기 위한 `docev_preview_sample.yaml` 설정 파일을 설명합니다.


## ⚙️ Configuration Breakdown

### 1. Paths & Output

- **`output_dir`**: 체크포인트와 로그가 저장되는 디렉토리

### 2. Model Arguments (`model`)

| Key                 | 설명                                                                                         |
|---------------------|--------------------------------------------------------------------------------------------|
| `_component_`       | 사용할 모델 컴포넌트 경로 (`torchtune.models.docev.docev_preview`)                          |
| `image_token_id`    | `<image>` 토큰에 예약된 ID (예: 64000)                                                      |
| `decoder_trainable` | 디코더 파라미터를 학습할지 여부                                                           |
| `encoder_trainable` | 인코더 파라미터를 학습할지 여부                                                           |
| `fusion_trainable`  | 비전과 텍스트를 연결하는 fusion 레이어를 학습할지 여부                                     |

### 3. Tokenizer & Transform (`tokenizer`)

| Key                           | 설명                                                                                                        |
|-------------------------------|-----------------------------------------------------------------------------------------------------------|
| `_component_`                 | 메시지를 모델 입력 형태로 변환하는 컴포넌트 (`torchtune.models.docev.docev_preview_transform`)          |
| `model_name_or_path`          | 사전학습 모델 경로 (`/app/docfm/checkpoints/release_models/docev-11.6b-32k-1.0.0-preview`)               |
| `image_token`                 | 이미지 자리표시자 토큰 (`"<image>"`)                                                                      |
| `stop_tokens`                 | 텍스트 생성을 중단할 토큰 목록                                                                              |
| `tile_size`, `patch_size`     | 비전 인코더 입력 이미지를 타일과 패치로 분할할 크기                                                       |
| `max_num_tiles`, `min_num_tiles` | 각 이미지당 사용할 최대/최소 타일 개수                                                                    |
| `vision_feature_select_strategy` | 특성 풀링 전략 (`full` 또는 `default`)                                                              |
| `sampling_ratio`              | connector 다운샘플링에 사용할 풀링 비율 목록 ([2,3])                                                       |
| `apply_random_sampling_ratio` | 각 배치마다 랜덤으로 풀링 비율을 적용할지 여부                                                            |
| `max_seq_len`                 | 토큰 + 이미지 토큰의 최대 시퀀스 길이 (8192)                                                               |
| `chat_template`               | 메시지와 이미지를 대화 형식으로 매핑하는 Jinja2 템플릿                                                     |

### 4. Checkpointer (`checkpointer`)

| Key                      | 설명                                                                                              |
|--------------------------|-------------------------------------------------------------------------------------------------|
| `_component_`            | 체크포인트 로딩/저장용 컴포넌트 (`torchtune.training.FullModelHFCheckpointer`)                 |
| `checkpoint_dir`         | 사전학습된 HF 모델 체크포인트 디렉토리                                                        |
| `checkpoint_files`       | safetensors 파일 이름 형식 및 최대 인덱스                                                      |
| `resume_from_checkpoint` | 이전 체크포인트로부터 재개할지 여부 (`False` = `model_name_or_path` 모델로 부터 새로 학습)                                        |

### 5. DataLoader (`dataloader`)

| Key                | 설명                                                                                           |
|--------------------|----------------------------------------------------------------------------------------------|
| `shuffle`          | 매 에포크마다 데이터셋을 무작위로 섞을지 여부                                                 |
| `collate_fn`       | 이미지+텍스트 배치를 위한 커스텀 collator 함수 (`padded_collate_tiled_images_and_mask`)     |
| `parallel_method`  | 데이터 로딩에 사용할 병렬화 방식 (`thread`)                                                   |
| `num_workers`      | 데이터 로딩 워커 프로세스 수 (4)                                                              |
| `pin_memory`       | CUDA로 전송 전 배치를 고정할지 여부                                                           |
| `packed`           | packed sequence 사용 여부 (`False` : packed는 현재 미지원)                                                         |
| `prefetch_factor`  | 각 워커가 미리 가져올 배치 수 (4)                                                            |

### 6. Datasets (`datasets`)

- **source**: `/app/docfm/datasets/public_dataset/vl3_syn7m/raw_ufx`
- **transform**: UFX 데이터를 TorchTune Message 포맷으로 변환하는 컴포넌트 (`_ufx.ufx_transform`)
- **weight**: 데이터셋 가중치 (1.0 = 전체 데이터 사용)

원시 샘플 형식은 [UFX Data Format](https://www.notion.so/UFX-481838e18be44a8cb3f1b4fd0725c08a?pvs=21)에서 확인하세요.

### 7. Fine-Tuning Arguments

| Key                         | 설명                                                            |
|-----------------------------|----------------------------------------------------------------|
| `epochs`                    | 전체 데이터셋 순회 횟수 (1회)                                  |
| `max_steps_per_epoch`       | 에포크당 최대 스텝 수 (`null` = 데이터셋 크기 기반)            |
| `batch_size`                | 배치 크기 (1; `현재 batch_size는 1만 가능` )                               |
| `gradient_accumulation_steps` | 그래디언트 누적 횟수 (32)                                     |
| `seed`                      | 랜덤 시드 (42)                                                 |

### 8. Optimization & Scheduling

| Component      | 설정                                                                       |
|----------------|---------------------------------------------------------------------------|
| `optimizer`    | `AdamW`, 학습률=1e-5, weight_decay=0.0, fused CUDA 커널 사용             |
| `lr_scheduler` | warmup 100 스텝 후 코사인 스케줄러                                        |
| `loss`         | 청크 출력 지원 CrossEntropy (`CEWithChunkedOutputLoss`)                   |
| `clip_grad_norm` | 그래디언트 클리핑 임계치 (1.0)                                           |
| `compile`      | `True` (`torch.compile` 사용)                                             |

### 9. Training Environment & Precision

- **device**: `cuda`
- **activation_checkpointing**: 비활성화
- **mixed precision**: `bf16`

### 10. Logging & Monitoring

| Key                         | 설명                                                         |
|-----------------------------|-------------------------------------------------------------|
| `metric_logger._component_` | Weights & Biases 로깅 컴포넌트                              |
| `project`                   | W&B 프로젝트 이름 (`torchtune`)                            |
| `group`                     | 실험 그룹 (`test`)                                         |
| `name`                      | 실험 이름 (`experiment-docev-preview-ufx-dataset`)        |
| `log_every_n_steps`         | 로그 기록 빈도 (스텝당 1회)                                 |
| `log_peak_memory_stats`     | GPU 메모리 사용량 기록 여부 (`True`)                       |
