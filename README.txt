!!Source code không chạy được trên python 3.11, source code đã test chạy ổn định trên python 3.10

Các bước cài đặt:
# 1. Giải nén folder dataset Oxford Building (5K) http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ vào thư mục souce code
# 2. Đảm bảo tất cả hình ảnh phải nằm trong thư mục ./oxbuild_images-v1
# 3. Cài đặt các thư viện bằng lệnh trên terminal:  pip install -r .\requirements.txt
# 4. Tải file features của dataset về bằng lệnh trên terminal: python .\download_features.py
# 4. Chạy web demo chạy lệnh trên terminal:  streamlit run main.py

Cấu trúc thư mục:
+ oxbuild_images-v1
	- all_souls_000000.jpg
	- all_souls_000001.jpg
	- all_souls_000002.jpg
	- all_souls_000003.jpg
	....
- main.py
- requirements.txt
- download_features.py
- delf_features_new.pkl
- imagesName.pkl
- README.txt