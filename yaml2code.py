import os
import datetime
import pyperclip
import yaml

def get_user_excluded_dirs():
    dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    print("Available directories:")
    for i, dir_name in enumerate(dirs, 1):
        print(f"{i}. {dir_name}")
    
    while True:
        try:
            user_input = input("Enter the numbers of directories to exclude (comma-separated) or press Enter to skip: ")
            if not user_input:
                return []
            excluded_indices = [int(x.strip()) - 1 for x in user_input.split(',')]
            return [dirs[i] for i in excluded_indices if 0 <= i < len(dirs)]
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

def process_directory(path, exclude_extensions=None, exclude_files=None, exclude_patterns=None, exclude_content_patterns=None, exclude_dirs=None):
    if exclude_extensions is None:
        exclude_extensions = [".pyc", ".pyo", ".exe", ".dll", ".so", ".o", ".a", ".bin", ".dat", ".jpeg"]
    if exclude_files is None:
        exclude_files = ["code2send_full.py", "code2text.txt", "code2send.py", "copy_to_clipboard.sh", "send2code.py"]
    if exclude_patterns is None:
        exclude_patterns = ["pack-"]
    if exclude_content_patterns is None:
        exclude_content_patterns = ["DIRC", "Q¨×6HÌ¦", "z÷ÈsxÚ", "lancedb", "ÿØÿà"]
    if exclude_dirs is None:
        exclude_dirs = []

    tree = {}
    for root, dirs, files in os.walk(path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Skip the Fastly_Opportunities directory
        if "Fastly_Opportunities" in root:
            continue

        current_level = tree
        for part in root.replace(path, "").split(os.sep):
            if part:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

        for filename in files:
            if any(filename.endswith(ext) for ext in exclude_extensions) or filename in exclude_files or any(pattern in filename for pattern in exclude_patterns):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as file:
                        content = file.read()
                except UnicodeDecodeError:
                    content = "[Binary file content not displayed]"

            if any(pattern in content for pattern in exclude_content_patterns):
                continue

            if "objects/" not in root:
                current_level[filename] = content
            else:
                current_level[filename] = "[Content not displayed for objects directory]"

    return tree

def main():
    root_directory = "."  # Replace with the root directory of your project
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"code2text_{timestamp}.yaml"
    
    exclude_dirs = get_user_excluded_dirs()
    tree = process_directory(root_directory, exclude_dirs=exclude_dirs)
    
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(tree, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Tree structure and file contents saved to {output_file}")

    # Copying the content of the output file to the clipboard
    with open(output_file, "r", encoding="utf-8") as file:
        content = file.read()
    pyperclip.copy(content)
    print(f"Content copied to clipboard")

    # Deleting the output file after copying content to clipboard
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"{output_file} has been deleted.")

if __name__ == "__main__":
    main()
