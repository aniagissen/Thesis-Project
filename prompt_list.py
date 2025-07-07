import os

prompt_root = "C:/Users/21006782/Prompts"
output_file = "C:/Users/21006782/prompts_list.txt"
count = 0

with open(output_file, "w", encoding="utf-8") as outfile:
    for scene_folder in sorted(os.listdir(prompt_root)):
        scene_path = os.path.join(prompt_root, scene_folder)
        if not os.path.isdir(scene_path):
            continue
        for file in sorted(os.listdir(scene_path)):
            if file.endswith(".json.txt"):
                full_path = os.path.join(scene_path, file)
                with open(full_path, "r", encoding="utf-8") as infile:
                    prompt = infile.read().strip().replace("\n", " ")
                    outfile.write(prompt + "\n")
                count += 1

print(f"Combined {count} prompts into {output_file}")
