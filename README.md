Try:
- Dynamic Mixing
- Freeze Encoder
- Dropout 

analyze_vivos.py -> Thực hiện quét tất cả tập train/test để thu thập thông tin về các file audio ngắn hơn 3s và silent để loại khỏi quá trình tạo dataset mixed. Kết quả được lưu vào file json.

Lệnh:
python -m src.analysis.quality_check

Future:
Refactor các file code trong src sử dụng configs chuẩn, truyền tham số lúc chạy file, load file json mới nhất

Target Project Structure:
speech_separation_project/  # Tên thư mục gốc của dự án
├── configs/                # Chứa các file config (YAML/JSON) cho hyperparameters, paths, etc.
│   ├── analysis.yaml       # Config cho analysis audio quality
│   ├── preprocessing.yaml  # Config cho preprocessing và pairing
│   ├── dataset.yaml        # Config cho tạo dataset
│   ├── training.yaml       # Config cho fine-tune Conv-TasNet
│   └── logging.yaml        # Config cho logging (e.g., mức độ log, paths)
├── data/                   # Chứa dữ liệu (không commit vào Git nếu dữ liệu lớn, dùng .gitignore hoặc DVC)
│   ├── raw/                # Dataset gốc (audio thô, không chỉnh sửa)
│   ├── processed/          # Audio sau preprocessing và filtering (loại bỏ không chất lượng)
│   └── datasets/           # Dataset cuối cùng (sau ghép cặp, sẵn sàng cho training)
│       ├── train/          # Dataset train
│       ├── val/            # Dataset validation
│       └── test/           # Dataset test
├── docs/                   # Tài liệu dự án
│   ├── README.md           # Hướng dẫn tổng quan, cách chạy dự án
│   ├── requirements.txt    # Danh sách dependencies (pip install -r requirements.txt)
│   └── architecture.md     # Mô tả kiến trúc dự án, flow dữ liệu
├── logs/                   # Chứa logs từ console và model (tự động generate)
│   ├── analysis_logs/      # Logs từ analysis
│   ├── training_logs/      # Logs từ fine-tune (e.g., TensorBoard, MLflow, hoặc text logs)
│   └── errors/             # Logs lỗi riêng biệt
├── models/                 # Chứa checkpoints model (Conv-TasNet sau fine-tune)
│   ├── pretrained/         # Model pretrained (nếu có)
│   └── finetuned/          # Checkpoints sau training (e.g., best_model.pth)
├── notebooks/              # Jupyter notebooks cho experiments/exploration
│   ├── analysis.ipynb      # Notebook cho analysis audio quality
│   ├── preprocessing.ipynb # Notebook cho thử nghiệm preprocessing và pairing
│   └── training.ipynb      # Notebook cho fine-tune model
├── src/                    # Code chính (core logic)
│   ├── analysis/           # Module cho analysis audio (tìm và loại bỏ audio kém chất lượng)
│   │   └── quality_check.py # Script kiểm tra chất lượng (e.g., SNR, silence detection)
│   ├── preprocessing/      # Module cho preprocessing và ghép cặp audio
│   │   ├── pair_audio.py   # Ghép cặp theo điều kiện (e.g., speaker, noise level)
│   │   └── preprocess.py   # Normalize, resample, augment audio
│   ├── dataset/            # Module cho tạo dataset từ data gốc
│   │   └── create_dataset.py # Tạo train/val/test từ processed data
│   ├── model/              # Module cho model Conv-TasNet
│   │   └── conv_tasnet.py  # Define model architecture, load pretrained nếu cần
│   ├── training/           # Module cho fine-tune và logging
│   │   ├── train.py        # Script fine-tune model
│   │   └── evaluate.py     # Đánh giá model
│   ├── utils/              # Các hàm hỗ trợ chung
│   │   ├── logger.py       # Setup logging cho console và file (e.g., dùng logging module hoặc WandB)
│   │   ├── audio_utils.py  # Hàm xử lý audio (load/save, metrics)
│   │   └── config_loader.py # Load configs
│   └── __init__.py         # Để src là package Python
├── tests/                  # Unit tests (dùng pytest)
│   ├── test_analysis.py    # Test cho analysis
│   ├── test_preprocessing.py # Test cho preprocessing
│   └── test_model.py       # Test cho model
├── .gitignore              # Ignore files lớn như data/raw, models/checkpoints
├── setup.py                # Nếu muốn package dự án (tùy chọn)
└── main.py                 # Entry point chính để chạy toàn bộ pipeline (e.g., python main.py --stage analysis)
