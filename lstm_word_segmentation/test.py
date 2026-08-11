from pathlib import Path

categories = ["news", "encyclopedia", "article", "novel"]
texts_range = range(90, 97)

base_dir = Path(__file__).parent.parent.absolute()
out_path = base_dir / "collated_90_96.txt"

with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for cat in categories:
        for n in texts_range:
            text_num_str = str(n)
            in_path = base_dir / "Data" / "Best" / cat / f"{cat}_000{text_num_str}.txt"

            if not in_path.exists():
                # skip missing files (or raise if you prefer)
                print('what')

            out.write(in_path.read_text(encoding="utf-8"))
