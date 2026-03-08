import json

f_out = open("conversation_dataset.jsonl", "w", encoding="utf-8")
with open("../E-commerce dataset/dev.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip().replace(" ","").split("	")[1:]
        contents = []
        for id,content in enumerate(line):
            if id%2 == 0:
                contents.append({"role": "user", "content": content})
            else:
                contents.append({"assistant": "user", "content": content})

        conversations = contents
        # 写入JSON格式
        json_line = json.dumps(
            {"conversations": conversations},
            ensure_ascii=False
        )
        f_out.write(json_line + "\n")
f_out.close()


