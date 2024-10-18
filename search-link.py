import google.generativeai as genai
import pandas as pd
import time

def run_program():
    # Cấu hình API key cho Google Generative AI
    genai.configure(api_key="AIzaSyClhiCnO3vjOPCcuQrkG2JZQEddvECfcqw")  # Thay bằng API key hợp lệ

    # Cấu hình mô hình và các tham số
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        # Các cấu hình an toàn nếu có
    ]

    # Khởi tạo mô hình Generative AI
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Đọc dữ liệu từ file
    df = pd.read_excel('Data-2.xlsx')

    # Tạo cột mới để lưu link hội nghị nếu chưa có
    if 'Conference_Link' not in df.columns:
        df['Conference_Link'] = ''

    # Tìm các hàng chưa có link hội nghị
    empty_link_rows = df[df['Conference_Link'].isna() | (df['Conference_Link'] == '')]

    # Số lượng hàng cần xử lý (có thể tăng lên nếu cần)
    num_rows_to_process = 5
    for i, row in empty_link_rows.head(num_rows_to_process).iterrows():
        title = row['Title']
        prompt = f"Find the official website for the conference titled: {title}."

        try:
            # Yêu cầu Generative AI tạo nội dung
            response = model.generate_content([prompt])
            link = response.text.strip()
            
            # Kiểm tra xem link trả về có hợp lệ không
            if link.startswith("http"):
                df.at[i, 'Conference_Link'] = link
            else:
                print(f"Không tìm thấy link hợp lệ cho tiêu đề '{title}'.")
                df.at[i, 'Conference_Link'] = "Link not found"

        except Exception as e:
            print(f"Error processing title '{title}': {e}")
            df.at[i, 'Conference_Link'] = "Error"

    # Lưu dữ liệu trở lại file
    df.to_excel('Data-2.xlsx', index=False)

    print(f"Cập nhật thành công {num_rows_to_process} hội nghị với link website.")

while True:
    run_program()
    print("Chờ 10 giây để chạy lại chương trình...")
    time.sleep(10)
