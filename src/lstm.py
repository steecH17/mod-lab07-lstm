with open("src/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    word_count = len(text.split())
    print(f"Слов в файле: {word_count}")