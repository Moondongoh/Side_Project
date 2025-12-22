import os

def generate_folder_structure(start_path, output_file):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)

        if level == 0:
            output_file.write(f"📁 {os.path.basename(root)}/\n")
        else:
            output_file.write(f"{indent}📁 {os.path.basename(root)}/\n")

        sub_indent = ' ' * 4 * (level + 1)

        for f in files:
            output_file.write(f"{sub_indent}📄 {f}\n")

if __name__ == "__main__":
    current_directory = os.getcwd()
    output_filename = "folder_structure.txt"

    print(f"'{current_directory}'의 폴더 구조를 '{output_filename}' 파일로 저장합니다...")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            generate_folder_structure(current_directory, f)
        print("✅ 작업이 완료되었습니다!")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")