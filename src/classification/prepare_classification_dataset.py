import argparse
import csv
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    source_path: Path
    class_name: str
    session_id: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepara dataset de classificação de vacas com split sem vazamento por session_id: "
            "10% teste estratificado e 5 folds (train/val) no restante."
        )
    )
    parser.add_argument(
        "--input-root",
        default="data/fotos_classificar",
        help="Diretório com imagens originais (preferencialmente em subpastas por classe).",
    )
    parser.add_argument(
        "--output-root",
        default="data/datasets/classifications",
        help="Diretório de saída para treino e avaliação de classificação.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Percentual para split de teste (padrão: 0.10).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Quantidade de folds train/val no conjunto de desenvolvimento.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade.",
    )
    parser.add_argument(
        "--session-regex",
        default="",
        help=(
            "Regex opcional para extrair session_id do nome do arquivo. "
            "Se houver grupo capturado, usa o grupo 1; caso contrário, usa o match inteiro."
        ),
    )
    parser.add_argument(
        "--use-symlinks",
        action="store_true",
        help="Cria links simbólicos ao invés de copiar imagens.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove o diretório de saída antes de gerar o novo dataset.",
    )
    return parser.parse_args()


def infer_session_id(file_path: Path, session_regex: str = "") -> str:
    stem = file_path.stem

    if session_regex:
        match = re.search(session_regex, stem)
        if match:
            if match.groups():
                return match.group(1)
            return match.group(0)

    tokens = stem.split("_")
    if len(tokens) >= 4 and re.fullmatch(r"\d{8}", tokens[0]) and re.fullmatch(r"\d{6}", tokens[1]):
        return "_".join(tokens[:4])
    if len(tokens) >= 2 and re.fullmatch(r"\d{8}", tokens[0]) and re.fullmatch(r"\d{6}", tokens[1]):
        return "_".join(tokens[:2])
    if len(tokens) >= 3:
        return "_".join(tokens[:3])
    return stem


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def load_samples(input_root: Path, session_regex: str):
    class_dirs = sorted([d for d in input_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    samples = []

    if class_dirs:
        for class_dir in class_dirs:
            images = list_images(class_dir)
            for image_path in images:
                samples.append(
                    Sample(
                        source_path=image_path,
                        class_name=class_dir.name,
                        session_id=infer_session_id(image_path, session_regex),
                    )
                )
    else:
        images = list_images(input_root)
        for image_path in images:
            class_name = image_path.stem.split("_")[0]
            samples.append(
                Sample(
                    source_path=image_path,
                    class_name=class_name,
                    session_id=infer_session_id(image_path, session_regex),
                )
            )

    if not samples:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {input_root}")

    df = pd.DataFrame(
        {
            "source_path": [str(s.source_path.resolve()) for s in samples],
            "class_name": [s.class_name for s in samples],
            "session_id": [s.session_id for s in samples],
        }
    )

    classes_sorted = sorted(df["class_name"].unique().tolist())
    class_to_index = {name: idx for idx, name in enumerate(classes_sorted)}
    df["class_index"] = df["class_name"].map(class_to_index)
    return df, class_to_index


def choose_best_test_split(df: pd.DataFrame, test_size: float, seed: int):
    y = df["class_name"].values
    groups = df["session_id"].values
    target_size = int(round(len(df) * test_size))

    if target_size <= 0:
        raise ValueError("test_size muito pequeno para o volume de dados.")

    n_splits_test = max(2, int(round(1.0 / test_size)))
    splitter = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)

    candidates = []
    for split_id, (_, test_idx) in enumerate(splitter.split(df, y=y, groups=groups)):
        candidates.append((split_id, test_idx, abs(len(test_idx) - target_size)))

    if not candidates:
        fallback = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        _, test_idx = next(fallback.split(df, y=y, groups=groups))
        return test_idx

    _, best_test_idx, _ = min(candidates, key=lambda item: item[2])
    return best_test_idx


def materialize_split(df_split: pd.DataFrame, destination_root: Path, use_symlinks: bool):
    for _, row in df_split.iterrows():
        src = Path(row["source_path"])
        class_name = row["class_name"]
        dst_dir = destination_root / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if use_symlinks:
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_split_stats(title: str, split_df: pd.DataFrame):
    class_counts = Counter(split_df["class_name"].tolist())
    sessions = split_df["session_id"].nunique()
    print(f"\n[{title}]")
    print(f"- imagens: {len(split_df)}")
    print(f"- classes: {len(class_counts)}")
    print(f"- sessões únicas: {sessions}")
    print("- distribuição por classe:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  - {class_name}: {count}")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_root}")

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    df, class_to_index = load_samples(input_root, args.session_regex)

    print("=" * 70)
    print("Preparação do dataset de classificação")
    print(f"Entrada: {input_root.resolve()}")
    print(f"Saída:   {output_root.resolve()}")
    print(f"Total de imagens: {len(df)}")
    print(f"Total de classes: {len(class_to_index)}")
    print("=" * 70)

    test_idx = choose_best_test_split(df, args.test_size, args.seed)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    dev_df = df.drop(index=df.index[test_idx]).copy().reset_index(drop=True)

    test_sessions = set(test_df["session_id"].tolist())
    dev_sessions = set(dev_df["session_id"].tolist())
    leaked_sessions = test_sessions.intersection(dev_sessions)
    if leaked_sessions:
        raise RuntimeError(
            f"Vazamento detectado entre teste e desenvolvimento. Sessões repetidas: {sorted(leaked_sessions)[:5]}"
        )

    materialize_split(
        test_df,
        destination_root=output_root / "test" / "images",
        use_symlinks=args.use_symlinks,
    )

    fold_rows = []
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold_id, (train_idx, val_idx) in enumerate(
        sgkf.split(dev_df, y=dev_df["class_name"].values, groups=dev_df["session_id"].values)
    ):
        train_df = dev_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = dev_df.iloc[val_idx].copy().reset_index(drop=True)

        train_sessions = set(train_df["session_id"].tolist())
        val_sessions = set(val_df["session_id"].tolist())
        leak = train_sessions.intersection(val_sessions)
        if leak:
            raise RuntimeError(f"Vazamento detectado no fold {fold_id}: {sorted(leak)[:5]}")

        fold_root = output_root / f"fold_{fold_id}"
        materialize_split(train_df, fold_root / "train" / "images", args.use_symlinks)
        materialize_split(val_df, fold_root / "val" / "images", args.use_symlinks)

        print_split_stats(f"fold_{fold_id} | train", train_df)
        print_split_stats(f"fold_{fold_id} | val", val_df)

        for split_name, split_df in (("train", train_df), ("val", val_df)):
            for _, row in split_df.iterrows():
                fold_rows.append(
                    {
                        "source_path": row["source_path"],
                        "class_name": row["class_name"],
                        "class_index": int(row["class_index"]),
                        "session_id": row["session_id"],
                        "fold": fold_id,
                        "split": split_name,
                    }
                )

    print_split_stats("teste estratificado", test_df)

    for _, row in test_df.iterrows():
        fold_rows.append(
            {
                "source_path": row["source_path"],
                "class_name": row["class_name"],
                "class_index": int(row["class_index"]),
                "session_id": row["session_id"],
                "fold": -1,
                "split": "test",
            }
        )

    save_csv(
        output_root / "splits_manifest.csv",
        rows=fold_rows,
        fieldnames=["source_path", "class_name", "class_index", "session_id", "fold", "split"],
    )
    save_csv(
        output_root / "classes.csv",
        rows=[{"class_name": name, "class_index": idx} for name, idx in class_to_index.items()],
        fieldnames=["class_name", "class_index"],
    )

    print("\nArquivos gerados:")
    print(f"- {output_root / 'splits_manifest.csv'}")
    print(f"- {output_root / 'classes.csv'}")
    print("\nEstrutura pronta para treino:")
    print(f"- {output_root / 'test' / 'images'}")
    for fold_id in range(args.n_splits):
        print(f"- {output_root / f'fold_{fold_id}' / 'train' / 'images'}")
        print(f"- {output_root / f'fold_{fold_id}' / 'val' / 'images'}")


if __name__ == "__main__":
    main()
