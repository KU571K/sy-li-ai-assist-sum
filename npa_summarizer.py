"""
Обработка нормативно-правовых актов (.md) и тезисное резюме с помощью OpenAI

Данный скрипт:
- читает NPA файл в формате `.md`
- нормализует и разбивает его на статьи/главы
- отправляет секции в LLM OpenAI
- сохраняет тезисный вариант НПА в отдельный файл
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv


# Загружаем переменные окружения из .env файла (если есть)
load_dotenv()

# Получаем API ключ из переменной окружения
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY не установлен. "
        "Установите переменную окружения или создайте файл .env с OPENAI_API_KEY=your_key"
    )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.artemox.com/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "npa-summarizer"
    }
)

SECTION_RX = re.compile(
    r'''(?mx)
    ^\s*(
        \#{1,6}\s+.+
        |(Глава|Статья|Раздел|Параграф|Часть)\s+
         ([IVXLC]+|\d+)\.?
         (?:\s*[-–.]?\s*.+)?
        |Раздел\s+[IVXLC]+(?:\s*[-–.]?\s*.+)?
    )\s*$
    ''',
)


def parse_text_file(path: str) -> str:
    """Читает и нормализует текстовый файл."""
    with open(path, 'r', encoding='utf-8') as f:
        t = f.read()
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t


def normalize_text(text: str) -> str:
    """Нормализует текст: убирает лишние пробелы и переносы строк."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(
        r'((Глава|Статья|Раздел|Параграф|Часть)\s+[IVXLC\d.]+)\s*\n\s*(\S)',
        r'\1 \3',
        text
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Разбивает текст на секции по заголовкам."""
    text = normalize_text(text)
    matches = list(SECTION_RX.finditer(text))
    if not matches:
        return [("document", text.strip())]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))

    return sections


def summarize_section(title: str, body: str) -> str:
    """Создает тезисное резюме секции документа с помощью LLM."""
    prompt = f"""
Ты — юридический аналитик.
Сделай краткий тезисный конспект следующей части нормативно-правового акта.
Сохраняй юридическую точность, формальный стиль и структуру.

Раздел: {title}

Текст:
{body}
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",  
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def summarize_document(md_path: str) -> str:
    """Обрабатывает документ и возвращает суммаризацию."""
    raw_text = parse_text_file(md_path)
    sections = split_into_sections(raw_text)

    summaries = []
    for i, (title, body) in enumerate(sections, 1):
        print(f"Обработка секции {i}/{len(sections)}: {title[:50]}...")
        summary = summarize_section(title, body)
        summaries.append(f"## {title}\n{summary}")

    return "\n\n".join(summaries)


def save_summary(md_path: str, summary: str, output_dir: str = None) -> str:
    """
    Сохраняет суммаризацию в отдельный файл.
    
    Args:
        md_path: Путь к исходному файлу
        summary: Текст суммаризации
        output_dir: Директория для сохранения (если None, то рядом с исходным файлом)
    
    Returns:
        Путь к сохраненному файлу
    """
    source_path = Path(md_path)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{source_path.stem}_summary.md"
    else:
        output_file = source_path.parent / f"{source_path.stem}_summary.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return str(output_file)


def process_file(md_path: str, output_dir: str = None) -> str:
    """
    Обрабатывает один файл и сохраняет результат.
    
    Args:
        md_path: Путь к файлу для обработки
        output_dir: Директория для сохранения результатов
    
    Returns:
        Путь к сохраненному файлу с суммаризацией
    """
    print(f"\nОбработка файла: {md_path}")
    print("=" * 60)
    
    summary = summarize_document(md_path)
    output_file = save_summary(md_path, summary, output_dir)
    
    print(f"\nРезультат сохранен в: {output_file}")
    return output_file


def collect_md_files(dir_path: str) -> List[str]:
    """Собирает пути ко всем .md файлам в директории (рекурсивно)."""
    base = Path(dir_path)
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.rglob("*.md")):
        if p.is_file():
            out.append(str(p))
    return out


def process_folder(input_dir: str, output_dir: str) -> List[str]:
    """
    Обрабатывает все .md файлы из папки и сохраняет суммаризации в отдельную папку.
    
    Args:
        input_dir: Папка с исходными .md файлами (напр. files/)
        output_dir: Папка для сохранения результатов (напр. summaries/)
    
    Returns:
        Список путей к сохраненным файлам
    """
    files = collect_md_files(input_dir)
    if not files:
        print(f"В папке '{input_dir}' не найдено .md файлов.")
        return []
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Найдено файлов: {len(files)}. Результаты → {output_dir}")
    
    saved = []
    failed = []
    for i, md_path in enumerate(files, 1):
        name = Path(md_path).name
        print(f"\n[{i}/{len(files)}] {name}")
        try:
            out = process_file(md_path, output_dir)
            saved.append(out)
        except Exception as e:
            print(f"  Ошибка: {e}")
            failed.append((md_path, str(e)))
    
    if failed:
        print(f"\nНе удалось обработать {len(failed)} файл(ов):")
        for path, err in failed:
            print(f"  - {Path(path).name}: {err}")
    print(f"\nУспешно обработано: {len(saved)} из {len(files)}.")
    return saved


def main():
    """Главная функция для запуска из командной строки."""
    if len(sys.argv) < 2:
        print("Использование:")
        print(f"  python {sys.argv[0]} <файл.md | папка/> [папка_для_сохранения]")
        print("\nПримеры:")
        print(f"  python {sys.argv[0]} files/law.md")
        print(f"  python {sys.argv[0]} files/law.md summaries/")
        print(f"  python {sys.argv[0]} files/ summaries/   # все .md из files/ → summaries/")
        sys.exit(1)
    
    src = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(src):
        print(f"Ошибка: '{src}' не найден.")
        sys.exit(1)
    
    try:
        if os.path.isdir(src):
            out = output_dir or "summaries"
            process_folder(src, out)
        else:
            process_file(src, output_dir)
        print("\nОбработка завершена.")
    except Exception as e:
        print(f"\nОшибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
