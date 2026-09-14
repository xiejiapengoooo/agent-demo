import json
import shutil
import subprocess
import sys
from pathlib import Path

from convert_mineru_content_list import convert_mineru_content_list_v2
from structure_aware import structure_aware

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "source"
MINERU_OUTPUT_DIR = BASE_DIR / "mineru-output"
DB_DIR = BASE_DIR / "db"
EMBED_MODEL_ID = "BAAI/bge-m3"
SKIP_TYPES = frozenset(
    {
        "page_header",
        "page_footer",
        "page_number",
        "page_aside_text",
        "page_footnote",
    }
)


def run_mineru():
    if MINERU_OUTPUT_DIR.exists():
        if not MINERU_OUTPUT_DIR.is_dir():
            raise NotADirectoryError(MINERU_OUTPUT_DIR)
        shutil.rmtree(MINERU_OUTPUT_DIR)
    MINERU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mineru.cli.client",
            "--path",
            str(SOURCE_DIR),
            "--output",
            str(MINERU_OUTPUT_DIR),
            "--backend",
            "hybrid-engine",
            "--method",
            "auto",
        ],
        check=True,
    )


if __name__ == "__main__":
    # source_files = sorted(SOURCE_DIR.rglob("*.pdf"))

    # if not source_files:
    #     raise FileNotFoundError(f"在 {SOURCE_DIR} 中没有找到 PDF 文件")

    # run_mineru()

    # mineru_output_files = sorted(MINERU_OUTPUT_DIR.rglob("*_content_list_v2.json"))
    # if not mineru_output_files:
    #     raise FileNotFoundError(
    #         f"MinerU 没有生成 *_content_list_v2.json：{MINERU_OUTPUT_DIR}"
    #     )

    pdf = (
        Path(__file__).resolve().parent
        / "mineru-output/浦发上海浦东发展银行西安分行个金客户经理考核办法/hybrid_auto/浦发上海浦东发展银行西安分行个金客户经理考核办法_content_list_v2.json"
    )
    payload = json.loads(pdf.read_text(encoding="utf-8"))
    converted = convert_mineru_content_list_v2(payload)
    sections = structure_aware(converted)
    print(json.dumps(sections, ensure_ascii=False))
